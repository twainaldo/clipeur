#!/bin/zsh
# Double-clic : démarre Clipeur et ouvre la page dans le navigateur.
# Pour l'arrêter : ferme cette fenêtre du Terminal (ou Ctrl+C).
cd "$(dirname "$0")" || exit 1
exec /usr/local/bin/python3 serveur.py --ouvrir
