"""Worker STT isolé.

Exécuté dans un processus séparé (via subprocess) plutôt qu'en tâche de fond
dans le serveur : l'inférence Whisper peut, sur certaines entrées, provoquer un
crash natif (segfault) impossible à intercepter par un try/except Python. En
l'isolant dans son propre processus, un tel crash termine uniquement ce
processus enfant (code de sortie négatif/anormal) au lieu d'emporter tout le
serveur applicatif et toutes les sessions en cours.

Usage :
    python stt_worker.py <audio_path> <language: fr|wo>

Sortie : une unique ligne JSON sur stdout.
    Succès : {"text": "..."}
    Échec  : {"error": "...", "error_type": "..."}
Tout le bruit de chargement de modèle (print/logging/warnings) est redirigé
vers stderr pour ne jamais polluer la ligne JSON de stdout.
"""
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "audio_path manquant", "error_type": "ValueError"}))
        return 1
    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # tout print() de chargement de modèle part sur stderr
    try:
        from input.stt import transcribe
        text = transcribe(audio_path, language=language)
        payload = {"text": text}
    except Exception as exc:  # noqa: BLE001 - on rapporte toute erreur au process parent
        payload = {"error": str(exc), "error_type": type(exc).__name__}
    finally:
        sys.stdout = real_stdout

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
