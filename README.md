# Clipeur

Site local pour Mac : une vidéo longue entre, des clips verticaux 1080 × 1920 sortent. Tout tourne sur la machine.

- Repère les moments forts d'une vidéo (son et image), demande un avis à Claude, puis fabrique les clips avec leur hook.
- Importe les vidéos de tweets et enregistre un direct Twitch par morceaux (`direct.py`).
- Pose un morceau sur un clip, calé sur son drop, avec les paroles en gros.
- Monte un « edit » sur un morceau (`edit.py`) : ouverture lente avec le titre, flash au drop, un plan par temps, fin au noir.

## Ce qu'il faut

- macOS et Python 3 avec `numpy` et `Pillow`.
- `ffmpeg` installé par Homebrew (`/opt/homebrew/bin`).
- Facultatif : la commande `claude` (Claude Code) pour l'avis sur les moments, `yt-dlp` pour le direct, un Python avec `faster-whisper` pour les paroles (son chemin se règle dans `PYTHONS_WHISPER`, au début de la partie « morceaux » de `serveur.py`).

## Lancer

Double-clic sur `lancer.command`, ou :

```
python3 serveur.py --ouvrir
```

Puis http://localhost:4848. `lancer.command` appelle `/usr/local/bin/python3` : adapte ce chemin si ton Python est ailleurs.

Les dossiers `entree/` (vidéos), `travail/` (analyses), `sorties/` (clips) et `sons/` (morceaux) se créent tout seuls. Ils ne sont pas dans le dépôt.

## Le monteur d'edits

```
python3 edit.py son morceau.mp3                      # tempo, grille des temps, drop
python3 edit.py monter edits/exemple.json sortie.mp4 # fabrique l'edit
```

Un plan de montage décrit le morceau, le dossier de plans, le titre et la liste des plans avec leur durée en temps (`1`, `2`, `0.5`, `0.25`). `edits/exemple.json` montre toutes les options courantes. `edit_mesure.py` décortique un edit existant : coupes, flashs, tempo, coupes posées sur les temps.

## L'extension Chrome

`chrome://extensions`, activer le mode développeur, « Charger l'extension non empaquetée », choisir le dossier `extension/`. Un bouton apparaît sous chaque tweet pour envoyer sa vidéo au site.
