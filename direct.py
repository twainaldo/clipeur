#!/usr/bin/env python3
"""
Clipeur, enregistreur de direct : suit un direct (Twitch, YouTube…) et le range sur ce Mac
par morceaux, que le site analyse au fur et à mesure.

Lancer :   /usr/local/bin/python3 direct.py https://www.twitch.tv/<chaîne> [minutes par morceau]
Arrêter :  Ctrl+C. Le morceau en cours est gardé.

Chaque morceau terminé arrive dans entree/ sous le nom direct_<chaîne>_<date>_<heure>.mp4.
Si le site Clipeur tourne, son analyse démarre toute seule ; sinon le morceau attend dans entree/.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

ICI = os.path.dirname(os.path.abspath(__file__))
ENTREE = os.path.join(ICI, "entree")
TAMPON = os.path.join(ICI, "travail", "direct")
SITE = "http://127.0.0.1:4848"


def trouver(nom, en_plus=()):
    for d in tuple(en_plus) + ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        p = os.path.join(d, nom)
        if os.path.exists(p):
            return p
    return nom


FFMPEG = trouver("ffmpeg")
YTDLP = trouver("yt-dlp", (os.path.expanduser("~/.local/bin"),))


def dire(texte):
    print(time.strftime("%H:%M:%S"), texte, flush=True)


def adresse_flux(lien):
    """L'adresse du flux vidéo du direct, ou rien s'il n'y a pas de direct en cours."""
    try:
        r = subprocess.run([YTDLP, "-g", "-f", "best", "--no-warnings", lien],
                           capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError):
        return ""
    lignes = [l for l in r.stdout.strip().splitlines() if l.startswith("http")]
    return lignes[-1] if r.returncode == 0 and lignes else ""


def confier(chemin):
    """Demande au site d'analyser le morceau. Renvoie le numéro de l'analyse, ou rien si le site dort."""
    try:
        req = urllib.request.Request(SITE + "/api/analyser",
                                     data=json.dumps({"chemin": chemin, "image": "rapide", "combien": 8}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r).get("id", "")
    except Exception:
        return ""


def finir(brut, chaine):
    """Un morceau terminé : on le range en MP4 propre dans entree/ et on le confie au site."""
    if os.path.getsize(brut) < 200000:      # moins d'une seconde d'image : rien à garder
        os.remove(brut)
        return
    quand = os.path.splitext(os.path.basename(brut))[0]          # AAAAMMJJ_HHMMSS
    cible = os.path.join(ENTREE, "direct_%s_%s.mp4" % (chaine, quand))
    r = subprocess.run([FFMPEG, "-y", "-v", "error", "-fflags", "+genpts", "-i", brut,
                        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", cible],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(cible) or os.path.getsize(cible) < 100000:
        dire("morceau gardé tel quel, sa mise en MP4 a échoué : " + brut)
        return
    os.remove(brut)
    numero = confier(cible)
    dire("morceau rangé : %s (%.0f Mo)%s" % (os.path.basename(cible), os.path.getsize(cible) / 1e6,
                                             ", analyse lancée" if numero else ", en attente dans entree/"))


def morceaux():
    return sorted(f for f in os.listdir(TAMPON) if f.endswith(".ts"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    lien = sys.argv[1]
    minutes = max(0.25, min(60.0, float(sys.argv[2]))) if len(sys.argv) > 2 else 10.0
    chaine = re.sub(r"[^A-Za-z0-9_]", "", lien.rstrip("/").split("/")[-1])[:30] or "direct"
    os.makedirs(ENTREE, exist_ok=True)
    os.makedirs(TAMPON, exist_ok=True)
    for f in morceaux():                     # des morceaux restés d'une fois précédente
        finir(os.path.join(TAMPON, f), chaine)

    arret = {"demande": False}

    def stop(*_):
        arret["demande"] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    vides, deja_vu = 0, False
    while not arret["demande"]:
        flux = adresse_flux(lien)
        if not flux:
            vides += 1
            if vides == 1:
                dire("pas de direct en cours sur %s, je regarde toutes les 30 s" % lien)
            # 10 minutes de silence après un direct : il est fini. Avant le premier : 2 heures d'attente.
            if vides > (20 if deja_vu else 240):
                break
            for _ in range(30):
                if arret["demande"]:
                    break
                time.sleep(1)
            continue
        vides, deja_vu = 0, True
        dire("direct trouvé, j'enregistre par morceaux de %g min" % minutes)
        journal = open(os.path.join(TAMPON, "ffmpeg.log"), "ab")
        proc = subprocess.Popen(
            [FFMPEG, "-v", "error", "-rw_timeout", "20000000", "-i", flux,
             "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
             "-f", "segment", "-segment_time", "%d" % round(minutes * 60), "-segment_format", "mpegts",
             "-reset_timestamps", "1", "-strftime", "1", os.path.join(TAMPON, "%Y%m%d_%H%M%S.ts")],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=journal)
        while proc.poll() is None and not arret["demande"]:
            time.sleep(5)
            for f in morceaux()[:-1]:        # tous sauf celui qui s'écrit encore
                finir(os.path.join(TAMPON, f), chaine)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # ffmpeg ferme proprement le morceau en cours
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        journal.close()
        for f in morceaux():
            finir(os.path.join(TAMPON, f), chaine)
        if not arret["demande"]:
            dire("le flux s'est arrêté, je regarde s'il reprend")
    dire("fin de l'enregistrement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
