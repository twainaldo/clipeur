"""Clipeur : les mots chantés d'un morceau et l'instant de chacun.

Lancé par le site avec un Python qui a faster-whisper (celui de Koda sur ce Mac), jamais avec le
Python du site lui-même. Écrit une seule ligne de JSON : {"langue", "duree", "mots": [{"d", "f", "m"}]}.
Tout se passe sur ce Mac, le modèle est déjà téléchargé.
"""

import json
import sys

from faster_whisper import WhisperModel

modele = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=8)
# une chanson : pas de filtre de silence (il couperait le chant), et chaque passage est écouté sans
# s'appuyer sur le précédent (sinon Whisper se met à répéter la même phrase)
segments, infos = modele.transcribe(sys.argv[1], vad_filter=False, word_timestamps=True,
                                    condition_on_previous_text=False)
mots = [{"d": round(w.start, 2), "f": round(w.end, 2), "m": w.word.strip()}
        for s in segments for w in (s.words or []) if w.word.strip()]
print(json.dumps({"langue": infos.language, "duree": round(infos.duration, 2), "mots": mots}))
