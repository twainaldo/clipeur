"""Décortique un edit de référence : coupes, flashs, rythme de la bande-son, et le lien entre les deux.
Usage : python3 -I analyse_edit.py <vidéo> <dossier de sortie>
Ne lit la vidéo qu'à travers ffmpeg / ffprobe. N'écrit que dans le dossier de sortie.
"""
import json
import os
import subprocess
import sys

import numpy as np

FF = "/opt/homebrew/bin/ffmpeg"
FP = "/opt/homebrew/bin/ffprobe"


def sonde(chemin):
    out = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", chemin],
                         capture_output=True, text=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return dict(w=int(s["width"]), h=int(s["height"]), fps=float(num) / float(den), dur=float(j["format"]["duration"]))


def images(chemin, w=96, h=54):
    raw = subprocess.run([FF, "-v", "error", "-i", chemin, "-vf", "scale=%d:%d:flags=area,format=gray" % (w, h),
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float32)


def son(chemin, sr=22050):
    raw = subprocess.run([FF, "-v", "error", "-i", chemin, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), sr


def coupes(im, fps):
    """Une coupe = un saut d'image net et isolé. Renvoie (instants, écart par image)."""
    d = np.abs(im[1:] - im[:-1]).mean(axis=(1, 2))
    d = np.concatenate([[0.0], d])
    out = []
    for i in range(1, len(d)):
        a, b = max(0, i - 12), min(len(d), i + 13)
        voisin = np.median(d[a:b])
        if d[i] > max(14.0, 3.5 * voisin + 4.0) and d[i] >= d[max(0, i - 2):i + 3].max():
            out.append(i)
    # deux « coupes » collées = un flash ou une secousse, on garde la première
    propres = []
    for i in out:
        if not propres or i - propres[-1] > 2:
            propres.append(i)
    return [i / fps for i in propres], d


def flashs(im, fps):
    """Image nettement plus claire que ses voisines pendant 1 à 3 images."""
    lum = im.mean(axis=(1, 2))
    out = []
    for i in range(2, len(lum) - 2):
        base = np.median(lum[max(0, i - 8):i + 9])
        if lum[i] - base > 45 and lum[i] >= lum[i - 2:i + 3].max():
            out.append(round(i / fps, 2))
    return out, lum


def spectre(y, sr, n=2048, saut=512):
    """Spectre de puissance par trames (numpy seul)."""
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    nb = 1 + (len(y) - n) // saut
    idx = np.arange(n)[None, :] + saut * np.arange(nb)[:, None]
    trames = y[idx] * np.hanning(n)[None, :]
    S = np.abs(np.fft.rfft(trames, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    t = (np.arange(nb) * saut + n / 2) / sr
    return S.T, f, t


def grille_des_temps(env, t, bpm_min=70.0, bpm_max=185.0):
    """Tempo fixe + phase : la grille régulière qui tombe le mieux sur les attaques (musique électronique)."""
    pas_t = float(t[1] - t[0])
    e = env - env.mean()
    meilleur = (None, -1e18)
    # recherche grossière par autocorrélation, puis fine sur la grille
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    cands = []
    for bpm in np.arange(bpm_min, bpm_max, 0.25):
        lag = 60.0 / bpm / pas_t
        i = int(lag)
        if i + 1 >= len(ac):
            continue
        v = ac[i] * (1 - (lag - i)) + ac[i + 1] * (lag - i)
        # on aime aussi que le double et la moitié du temps soient réguliers
        l2 = 2 * lag
        j = int(l2)
        if j + 1 < len(ac):
            v += 0.5 * (ac[j] * (1 - (l2 - j)) + ac[j + 1] * (l2 - j))
        poids = np.exp(-0.5 * (np.log2(bpm / 128.0) / 0.55) ** 2)
        cands.append((v * poids, bpm))
    cands.sort(reverse=True)
    for _, bpm0 in cands[:6]:
        for bpm in np.arange(bpm0 - 0.4, bpm0 + 0.4, 0.02):
            per = 60.0 / bpm
            for ph in np.arange(0.0, per, pas_t / 2):
                g = np.arange(ph, t[-1], per)
                v = float(np.interp(g, t, env).mean()) * np.exp(-0.5 * (np.log2(bpm / 128.0) / 0.55) ** 2)
                if v > meilleur[1]:
                    meilleur = ((bpm, ph), v)
    bpm, ph = meilleur[0]
    return float(bpm), np.arange(ph, t[-1], 60.0 / bpm)


def rythme(y, sr):
    S, f, t = spectre(y, sr)
    L = np.log1p(1000.0 * S / (S.max() + 1e-12))
    flux = np.maximum(0.0, L[:, 1:] - L[:, :-1]).sum(axis=0)
    env = np.concatenate([[0.0], flux])
    k = 3
    env = np.convolve(env, np.ones(k) / k, mode="same")
    tempo, tb = grille_des_temps(env, t)
    bas = S[f < 150].sum(axis=0)
    tous = S.sum(axis=0)
    return tempo, tb, t, bas, tous, env


def ecart_au_plus_proche(x, grille):
    if len(grille) == 0:
        return None
    return float(np.min(np.abs(np.asarray(grille) - x)))


def planches(chemin, dossier, nom, dur, par_seconde=4, colonnes=8, lignes=6, largeur=200):
    par_planche = colonnes * lignes
    total = int(dur * par_seconde) + 1
    n = (total + par_planche - 1) // par_planche
    sorties = []
    for k in range(n):
        debut = k * par_planche / par_seconde
        cible = os.path.join(dossier, "%s_planche_%02d.jpg" % (nom, k + 1))
        vf = ("fps=%d,scale=%d:-2,drawtext=text='%%{pts\\:hms}':x=4:y=4:fontsize=13:fontcolor=white:box=1:boxcolor=black@0.6,"
              "tile=%dx%d:padding=2:margin=2") % (par_seconde, largeur, colonnes, lignes)
        subprocess.run([FF, "-v", "error", "-y", "-ss", "%.3f" % debut, "-t", "%.3f" % (par_planche / par_seconde), "-i", chemin,
                        "-vf", vf, "-frames:v", "1", "-q:v", "4", cible], capture_output=True)
        if os.path.exists(cible):
            sorties.append(cible)
    return sorties


def main():
    chemin, dossier = sys.argv[1], sys.argv[2]
    os.makedirs(dossier, exist_ok=True)
    nom = "".join(c if c.isalnum() else "_" for c in os.path.splitext(os.path.basename(chemin))[0])[:40]
    info = sonde(chemin)
    im = images(chemin)
    fps = len(im) / info["dur"] if info["dur"] else info["fps"]
    tc, d = coupes(im, fps)
    tf, lum = flashs(im, fps)
    y, sr = son(chemin)
    res = dict(fichier=os.path.basename(chemin), **info, images=len(im), coupes=[round(t, 3) for t in tc], flashs=tf)
    if len(y) > sr:
        tempo, tb, t, bas, tous, env = rythme(y, sr)
        res["tempo"] = round(tempo, 1)
        res["temps"] = [round(float(x), 3) for x in tb]
        # demi-temps et quarts de temps
        if len(tb) > 2:
            pas = float(np.median(np.diff(tb)))
            demi = np.concatenate([tb, tb[:-1] + np.diff(tb) / 2])
            e_t = [ecart_au_plus_proche(c, tb) for c in tc]
            e_d = [ecart_au_plus_proche(c, demi) for c in tc]
            res["pas_du_temps_s"] = round(pas, 3)
            res["coupes_sur_un_temps_60ms"] = int(sum(1 for e in e_t if e is not None and e <= 0.06))
            res["coupes_sur_un_demi_temps_60ms"] = int(sum(1 for e in e_d if e is not None and e <= 0.06))
            plans = np.diff([0.0] + tc + [info["dur"]])
            res["plans"] = [round(float(p), 2) for p in plans]
            res["plan_median_s"] = round(float(np.median(plans)), 2)
            res["plan_median_en_temps"] = round(float(np.median(plans)) / pas, 2)
        # le drop : plus grosse montée de basses lissée
        k = max(1, int(0.5 * sr / 512))
        lisse = np.convolve(bas, np.ones(k) / k, mode="same")
        avant = np.concatenate([np.zeros(2 * k), lisse[:-2 * k]])
        montee = lisse - avant
        i = int(np.argmax(montee[2 * k:])) + 2 * k
        res["drop_vers_s"] = round(float(t[max(0, i - k)]), 2)
        # niveau sonore par seconde (relatif)
        par_s = [float(np.sqrt(np.mean(y[int(a * sr):int((a + 1) * sr)] ** 2))) for a in range(int(len(y) / sr))]
        m = max(par_s) or 1.0
        res["niveau_par_seconde"] = [round(v / m, 2) for v in par_s]
        bas_s = [float(np.mean(bas[(t >= a) & (t < a + 1)])) for a in range(int(t[-1]))]
        mb = max(bas_s) or 1.0
        res["basses_par_seconde"] = [round(v / mb, 2) for v in bas_s]
    res["luminosite_par_seconde"] = [round(float(lum[int(a * fps):int((a + 1) * fps)].mean()), 0) for a in range(int(info["dur"]))]
    res["agitation_par_seconde"] = [round(float(d[int(a * fps):int((a + 1) * fps)].mean()), 1) for a in range(int(info["dur"]))]
    res["planches"] = planches(chemin, dossier, nom, info["dur"])
    json.dump(res, open(os.path.join(dossier, nom + ".json"), "w"), ensure_ascii=False, indent=1)
    court = {k: v for k, v in res.items() if k not in ("temps",)}
    print(json.dumps(court, ensure_ascii=False))


if __name__ == "__main__":
    main()
