"""Monteur d'edits : pose des plans sur un morceau comme le font les edits les plus vus
(ouverture lente avec le titre, flash au drop, un plan par temps, fin au noir).

Usage :
  python3 -I monteur.py son <morceau> [drop approximatif en s]     -> tempo, grille, drop exact
  python3 -I monteur.py monter <plan.json> <sortie.mp4>            -> fabrique l'edit

Ne dépend que de numpy, Pillow et ffmpeg. N'écrit que dans le dossier de travail donné par le plan et dans la sortie.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

FF = "/opt/homebrew/bin/ffmpeg"
FP = "/opt/homebrew/bin/ffprobe"
IPS = 30                      # images par seconde de l'edit
LARG, HAUT = 1440, 1080       # image de travail (4:3)


def lancer(cmd):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg a échoué :\n" + r.stderr.decode("utf-8", "replace")[-1500:])
    return r


# ----------------------------------------------------------------------------- le son

def lire_son(chemin, sr=22050):
    raw = subprocess.run([FF, "-v", "error", "-i", chemin, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), sr


def spectre(y, sr, n=2048, saut=256):
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    nb = 1 + (len(y) - n) // saut
    idx = np.arange(n)[None, :] + saut * np.arange(nb)[:, None]
    S = np.abs(np.fft.rfft(y[idx] * np.hanning(n)[None, :], axis=1)) ** 2
    return S.T, np.fft.rfftfreq(n, 1.0 / sr), (np.arange(nb) * saut + n / 2) / sr


def analyser_son(chemin, drop_vers=None, bpm_min=100.0, bpm_max=180.0):
    """Tempo fixe + phase calés sur les attaques, puis le drop = le temps où les basses montent le plus."""
    y, sr = lire_son(chemin)
    S, f, t = spectre(y, sr)
    L = np.log1p(1000.0 * S / (S.max() + 1e-12))
    env = np.concatenate([[0.0], np.maximum(0.0, L[:, 1:] - L[:, :-1]).sum(axis=0)])
    # les attaques graves (grosse caisse) comptent double : c'est elles qu'on entend comme « le temps »
    Lb = L[f < 160]
    envb = np.concatenate([[0.0], np.maximum(0.0, Lb[:, 1:] - Lb[:, :-1]).sum(axis=0)])
    env = env / (env.max() + 1e-9) + 2.0 * envb / (envb.max() + 1e-9)
    pas_t = float(t[1] - t[0])
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    cands = []
    for bpm in np.arange(bpm_min, bpm_max, 0.1):
        v = 0.0
        for mult, poids in ((1, 1.0), (2, 0.6), (4, 0.4)):
            lag = mult * 60.0 / bpm / pas_t
            i = int(lag)
            if i + 1 < len(ac):
                v += poids * (ac[i] * (1 - (lag - i)) + ac[i + 1] * (lag - i))
        cands.append((v, bpm))
    cands.sort(reverse=True)
    meilleur = (None, -1e18)
    vus = []
    for _, bpm0 in cands:
        if any(abs(bpm0 - b) < 1.0 for b in vus):
            continue
        vus.append(bpm0)
        if len(vus) > 5:
            break
        for bpm in np.arange(bpm0 - 0.5, bpm0 + 0.5, 0.01):
            per = 60.0 / bpm
            for ph in np.arange(0.0, per, pas_t / 2):
                g = np.arange(ph, t[-1], per)
                v = float(np.interp(g, t, env).mean())
                if v > meilleur[1]:
                    meilleur = ((float(bpm), float(ph)), v)
    bpm, ph = meilleur[0]
    per = 60.0 / bpm
    temps = np.arange(ph, t[-1], per)
    bas = S[f < 150].sum(axis=0)

    def moyenne(a, b):
        m = (t >= a) & (t < b)
        return float(bas[m].mean()) if m.any() else 0.0

    # le drop = le premier temps où les basses reviennent POUR DE BON : plein sur les 4 temps qui suivent, vide juste avant
    ref = float(np.percentile(bas, 90)) + 1e-12
    scores = []
    for x in temps:
        if x < 2 * per or x > t[-1] - 5 * per:
            continue
        if drop_vers is not None and abs(x - drop_vers) > 1.2:
            continue
        apres = min(moyenne(x + k * per, x + (k + 1) * per) for k in range(4))
        avant = moyenne(x - per, x)
        scores.append(((apres - avant) / ref, float(x)))
    scores.sort(reverse=True)
    drop = scores[0][1] if scores else None
    return dict(bpm=round(bpm, 2), periode=per, phase=ph, duree=float(len(y) / sr), drop=drop,
                drops=[(round(x, 2), round(s, 2)) for s, x in scores[:4]])


# ----------------------------------------------------------------------------- les plans

def sonde(chemin):
    o = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate:format=duration", "-of", "json", chemin],
                       capture_output=True, text=True).stdout
    j = json.loads(o)
    s = j["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return dict(w=int(s["width"]), h=int(s["height"]), ips=float(a) / float(b), duree=float(j["format"]["duration"]))


def images_grises(chemin, w=96, h=54, debut=None, duree=None):
    cmd = [FF, "-v", "error"]
    if debut is not None:
        cmd += ["-ss", "%.3f" % debut]
    if duree is not None:
        cmd += ["-t", "%.3f" % duree]
    cmd += ["-i", chemin, "-vf", "scale=%d:%d:flags=area,format=gray" % (w, h), "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float32)


def zone_utile(chemin, info, debut=0.0, duree=None):
    """Rectangle sans les bandes noires incrustées (films en format large)."""
    w, h = 480, 270
    im = images_grises(chemin, w, h, debut, duree)
    if len(im) == 0:
        return 0, 0, info["w"], info["h"]
    mx = im[:: max(1, len(im) // 8)].max(axis=0)
    lignes = np.where(mx.max(axis=1) > 24)[0]
    cols = np.where(mx.max(axis=0) > 24)[0]
    if len(lignes) < h * 0.4 or len(cols) < w * 0.4:
        return 0, 0, info["w"], info["h"]
    fx, fy = info["w"] / w, info["h"] / h
    x0, x1 = int((cols[0]) * fx), int((cols[-1] + 1) * fx)
    y0, y1 = int((lignes[0] + 1) * fy), int((lignes[-1]) * fy)
    return x0, y0, x1 - x0, y1 - y0


def fenetre(chemin, info, duree_voulue, mode="fort", debut=0.0, fin=None):
    """Le meilleur passage continu d'un plan : le plus animé (« fort »), le plus posé (« calme ») ou le milieu."""
    fin = info["duree"] if fin is None else fin
    im = images_grises(chemin, debut=debut, duree=fin - debut)
    n = len(im)
    if n < 3:
        return debut, max(0.04, fin - debut)
    ips = n / (fin - debut)
    d = np.concatenate([[0.0], np.abs(im[1:] - im[:-1]).mean(axis=(1, 2))])
    med = float(np.median(d))
    sauts = [i for i in range(1, n) if d[i] > max(14.0, 3.5 * med + 4.0)]
    bornes = [0] + sauts + [n]
    a, b = max(zip(bornes[:-1], bornes[1:]), key=lambda p: p[1] - p[0])
    a, b = a + (1 if a else 0), b - (1 if b < n else 0)
    L = int(round(duree_voulue * ips))
    if b - a <= L:
        return debut + a / ips, max(0.04, (b - a) / ips)
    if mode == "milieu":
        i = a + (b - a - L) // 2
    else:
        c = np.concatenate([[0.0], np.cumsum(d)])
        sc = [c[i + L] - c[i] for i in range(a, b - L + 1)]
        i = a + (int(np.argmax(sc)) if mode == "fort" else int(np.argmin(sc)))
    return debut + i / ips, L / ips


def pair(x):
    return int(x) // 2 * 2


def segment(plan, n_images, sortie, dossier_plans, style):
    """Un plan de l'edit : recadré en 4:3, à la bonne vitesse, avec son entrée (flash, coup de zoom), exactement n images."""
    chemin = plan["fichier"] if os.path.isabs(plan.get("fichier", "")) else os.path.join(dossier_plans, plan["fichier"])
    info = sonde(chemin)
    tc = n_images / IPS
    vitesse = float(plan.get("vitesse", 1.0))
    if "vitesse" not in plan and not plan.get("rampe") and info["ips"] < IPS - 1 and IPS / info["ips"] <= 1.3 and plan.get("choix") != "calme":
        vitesse = IPS / info["ips"]
    rampe = plan.get("rampe")                       # exposant > 1 : part vite, arrive lentement
    source = tc * vitesse if not rampe else float(plan.get("source", tc * 1.6))
    d0, d1 = plan.get("de", 0.0), plan.get("a")
    if "debut" in plan:
        a, ds = float(plan["debut"]), source
    else:
        a, ds = fenetre(chemin, info, source, plan.get("choix", "fort"), d0, d1)
    if ds < source * 0.98:                          # plan trop court : on le ralentit pour tenir la durée
        vitesse = ds / tc
        rampe = None
    ux, uy, uw, uh = zone_utile(chemin, info, a, ds)
    ratio = LARG / HAUT
    cw = min(uw, uh * ratio)
    ch = cw / ratio
    xr, yr = float(plan.get("x", 0.5)), float(plan.get("y", 0.5))
    cx, cy = ux + (uw - cw) * xr, uy + (uh - ch) * yr
    f = ["crop=%d:%d:%d:%d" % (pair(cw), pair(ch), pair(cx), pair(cy))]
    if not rampe and vitesse < 0.97 and info["ips"] * vitesse < IPS - 0.5:
        f.append("minterpolate=fps=%d:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" % int(math.ceil(IPS / vitesse / 10.0) * 10))
    if rampe:
        f.append("setpts='(%.5f*pow(min(T/%.5f\\,1)\\,%.3f))/TB'" % (tc, ds, float(rampe)))
    elif abs(vitesse - 1.0) > 0.01:
        f.append("setpts=PTS/%.5f" % vitesse)
    if plan.get("flou") or (rampe and plan.get("flou", True)):
        f.append("tmix=frames=2:weights='1 1'")
    f.append("fps=%d" % IPS)
    zoom = plan.get("zoom")                         # « coup » (entrée qui cogne), « avant » / « arriere » (lent)
    if zoom in ("avant", "arriere"):
        amp = float(plan.get("zoom_amp", 0.07))
        f.append("scale=%d:%d:flags=lanczos" % (LARG * 2, HAUT * 2))
        z = ("1+%.4f*on/%d" % (amp, max(1, n_images - 1))) if zoom == "avant" else ("1+%.4f*(1-on/%d)" % (amp, max(1, n_images - 1)))
        f.append("zoompan=z='%s':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=%dx%d:fps=%d" % (z, LARG, HAUT, IPS))
    elif zoom == "coup":
        amp = float(plan.get("zoom_amp", 0.10))
        f.append("scale=%d:%d:flags=lanczos" % (LARG * 2, HAUT * 2))
        f.append("zoompan=z='1+%.4f*exp(-on/%.2f)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=%dx%d:fps=%d"
                 % (amp, float(plan.get("zoom_tau", 2.2)), LARG, HAUT, IPS))
    else:
        f.append("scale=%d:%d:flags=lanczos" % (LARG, HAUT))
    secousse = float(plan.get("secousse", 0.0))     # tremblement qui s'éteint, en pixels
    if secousse > 0:
        m = int(secousse) + 4
        f.append("scale=%d:%d" % (LARG + 2 * m, pair((LARG + 2 * m) * HAUT / LARG)))
        f.append("crop=%d:%d:x='(iw-%d)/2+%.1f*sin(2*PI*t*17)*exp(-t/0.12)':y='(ih-%d)/2+%.1f*cos(2*PI*t*13)*exp(-t/0.12)'"
                 % (LARG, HAUT, LARG, secousse, HAUT, secousse * 0.7))
    if plan.get("sombre"):
        f.append("eq=brightness=%.3f:saturation=%.3f:contrast=1.04" % (-float(plan["sombre"]), float(plan.get("saturation", 0.72))))
    if plan.get("flash") == "blanc":
        f.append("fade=t=in:st=0:d=%.3f:color=white" % float(plan.get("flash_duree", 0.22)))
    elif plan.get("flash"):
        f.append("eq=brightness='%.3f*exp(-t/0.06)':eval=frame" % float(plan["flash"]))
    f.append("format=yuv420p")
    lancer([FF, "-v", "error", "-y", "-ss", "%.4f" % a, "-t", "%.4f" % (ds + 0.25), "-i", chemin, "-an",
            "-vf", ",".join(f), "-frames:v", str(n_images), "-r", str(IPS),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", sortie])
    return dict(fichier=os.path.basename(chemin), debut=round(a, 2), source=round(ds, 2), images=n_images, vitesse=round(vitesse, 2))


def noir(n_images, sortie):
    lancer([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=%dx%d:r=%d" % (LARG, HAUT, IPS), "-frames:v", str(n_images),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", sortie])


# ----------------------------------------------------------------------------- le titre

POLICES = {
    "cinema": ("/System/Library/Fonts/Supplemental/Didot.ttc", 2),
    "affiche": ("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf", 0),
    "fine": ("/System/Library/Fonts/Supplemental/Futura.ttc", 0),
}


def titre_png(texte, visibles, sous_titre, sortie, cadre, police="cinema", taille=150, couleur=(245, 242, 235),
              or_=(216, 180, 106), espace=0.16, montrer_sous_titre=True, hauteur=0.5):
    """Le titre posé au milieu de l'image, `visibles` lettres affichées ; chaque lettre garde sa place définitive."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    W, H, iy, ih = cadre
    chemin, index = POLICES[police]
    f = ImageFont.truetype(chemin, taille, index=index)
    largeurs = [f.getlength(c) for c in texte]
    esp = taille * espace
    total = sum(largeurs) + esp * (len(texte) - 1)
    if total > W * 0.86:                             # titre long : on réduit pour qu'il tienne
        k = W * 0.86 / total
        return titre_png(texte, visibles, sous_titre, sortie, cadre, police, int(taille * k), couleur, or_, espace, montrer_sous_titre, hauteur)
    calque = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ombre = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d, do = ImageDraw.Draw(calque), ImageDraw.Draw(ombre)
    haut, bas = f.getbbox("H")[1], f.getbbox("H")[3]
    y = iy + ih * hauteur - (bas + haut) / 2 - (taille * 0.18 if sous_titre else 0)
    x = (W - total) / 2
    vus = 0
    for c, lw in zip(texte, largeurs):
        if c != " ":
            vus += 1
        if vus <= visibles and c != " ":
            do.text((x, y + 3), c, font=f, fill=(0, 0, 0, 190))
            d.text((x, y), c, font=f, fill=couleur + (255,))
        x += lw + esp
    if sous_titre and montrer_sous_titre:
        fs = ImageFont.truetype(POLICES["fine"][0], max(26, int(taille * 0.27)), index=POLICES["fine"][1])
        e2 = fs.size * 0.36
        l2 = [fs.getlength(c) for c in sous_titre]
        t2 = sum(l2) + e2 * (len(sous_titre) - 1)
        x2, y2 = (W - t2) / 2, y + bas + taille * 0.30
        for c, lw in zip(sous_titre, l2):
            do.text((x2, y2 + 2), c, font=fs, fill=(0, 0, 0, 190))
            d.text((x2, y2), c, font=fs, fill=or_ + (255,))
            x2 += lw + e2
    ombre = ombre.filter(ImageFilter.GaussianBlur(7))
    Image.alpha_composite(ombre, calque).save(sortie)


# ----------------------------------------------------------------------------- l'assemblage

ETALONNAGES = {
    # chaud dans les clairs, froid dans les ombres, contraste et couleurs poussés : le rendu des edits de films les plus vus
    "film": "eq=contrast=1.14:saturation=1.36:gamma=1.13,colorbalance=rs=-0.05:bs=0.07:rm=0.02:bm=-0.015:rh=0.07:gh=0.02:bh=-0.08,"
            "unsharp=5:5:0.6:5:5:0.0,vignette=angle=PI/7",
    # noir et blanc dense : le rendu de l'edit de foot le plus vu
    "foot": "hue=s=0,eq=contrast=1.32:brightness=-0.035:gamma=0.94,unsharp=5:5:0.8:5:5:0.0,vignette=angle=PI/5",
    # couleurs gardées, sombre et contrasté
    "sombre": "eq=contrast=1.22:saturation=0.82:brightness=-0.03:gamma=0.93,colorbalance=bs=0.05:bh=0.02,unsharp=5:5:0.7:5:5:0.0,vignette=angle=PI/5",
    "aucun": "null",
}


FORMATS = {"4:3": (1440, 1080), "3:4": (1080, 1440), "1:1": (1080, 1080), "9:16": (1080, 1920), "16:9": (1920, 1080)}


def monter(plan_json, sortie):
    global LARG, HAUT, IPS
    p = json.load(open(plan_json, encoding="utf-8"))
    LARG, HAUT = FORMATS[p.get("format", "4:3")]
    IPS = int(p.get("ips", 30))
    travail = p["travail"]
    os.makedirs(travail, exist_ok=True)
    son = analyser_son(p["son"], p.get("drop_vers"))
    per = son["periode"]
    drop = p.get("drop", son["drop"])
    temps_avant = float(p.get("temps_avant", 8))
    plans = p["plans"]
    total_temps = sum(float(x["temps"]) for x in plans)
    debut_son = drop - temps_avant * per
    if debut_son < 0:
        raise SystemExit("Pas assez de musique avant le drop pour %s temps d'ouverture." % temps_avant)
    # bornes en images, calées sur la grille des temps (pas de dérive)
    cumul, bornes = 0.0, [0]
    for x in plans:
        cumul += float(x["temps"])
        bornes.append(int(round(cumul * per * IPS)))
    liste, journal = [], []
    for i, x in enumerate(plans):
        n = bornes[i + 1] - bornes[i]
        seg = os.path.join(travail, "seg_%02d.mp4" % i)
        if x.get("noir"):
            noir(n, seg)
            journal.append(dict(noir=True, images=n))
        else:
            journal.append(segment(x, n, seg, p["plans_dossier"], p.get("style", "film")))
        journal[-1]["t"] = round(bornes[i] / IPS, 3)
        liste.append(seg)
    with open(os.path.join(travail, "liste.txt"), "w") as fl:
        for s in liste:
            fl.write("file '%s'\n" % s)
    duree = bornes[-1] / IPS
    # mise en page verticale : l'image 4:3 au milieu d'un cadre 1080 x 1920
    W, H = 1080, 1920
    ih = int(W * HAUT / LARG)
    iy = int(p.get("image_y", max(0, (H - ih) // 2 - (40 if ih < 1200 else 0))))
    entrees = ["-f", "concat", "-safe", "0", "-i", os.path.join(travail, "liste.txt"),
               "-ss", "%.4f" % debut_son, "-t", "%.4f" % duree, "-i", p["son"]]
    filtres = ["[0:v]%s,scale=%d:%d:flags=lanczos,pad=%d:%d:0:%d:black,setsar=1[v0]" % (ETALONNAGES[p.get("etalonnage", "film")], W, ih, W, H, iy)]
    courant, k = "v0", 2
    titre = p.get("titre")
    if titre:
        lettres = [c for c in titre["texte"] if c != " "]
        t0 = float(titre.get("depuis_temps", 1)) * per
        pas = float(titre.get("pas_temps", 1)) * per
        fin_titre = int(round(float(titre.get("jusqua_temps", temps_avant)) * per * IPS)) / IPS
        for n in range(1, len(lettres) + 1):
            a = t0 + (n - 1) * pas
            b = fin_titre if n == len(lettres) else t0 + n * pas
            if a >= fin_titre:
                break
            png = os.path.join(travail, "titre_%02d.png" % n)
            titre_png(titre["texte"], n, titre.get("sous_titre"), png, (W, H, iy, ih), titre.get("police", "cinema"),
                      int(titre.get("taille", 150)), montrer_sous_titre=(n >= int(titre.get("sous_titre_a", 1))),
                      hauteur=float(titre.get("hauteur", 0.5)))
            entrees += ["-loop", "1", "-t", "%.3f" % duree, "-i", png]
            filtres.append("[%s][%d:v]overlay=0:0:enable='between(t,%.4f,%.4f)'[v%d]" % (courant, k, a, b - 0.5 / IPS, k))
            courant = "v%d" % k
            k += 1
    filtres.append("[1:a]afade=t=in:st=0:d=0.03,afade=t=out:st=%.3f:d=0.10,aresample=48000[a]" % (duree - 0.10))
    lancer([FF, "-v", "error", "-y"] + entrees + ["-filter_complex", ";".join(filtres), "-map", "[%s]" % courant, "-map", "[a]",
            "-t", "%.4f" % duree, "-r", str(IPS), "-c:v", "libx264", "-preset", "medium", "-crf", str(p.get("crf", 19)), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", sortie])
    bilan = dict(sortie=sortie, duree=round(duree, 3), bpm=son["bpm"], drop_dans_le_morceau=round(drop, 3), debut_dans_le_morceau=round(debut_son, 3),
                 drop_dans_l_edit=round(temps_avant * per, 3), temps=total_temps, plans=journal)
    json.dump(bilan, open(os.path.join(travail, "bilan.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in bilan.items() if k != "plans"}, ensure_ascii=False))
    for j in journal:
        print("  ", j)


if __name__ == "__main__":
    if sys.argv[1] == "son":
        print(json.dumps(analyser_son(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None), ensure_ascii=False))
    elif sys.argv[1] == "monter":
        monter(sys.argv[2], sys.argv[3])
