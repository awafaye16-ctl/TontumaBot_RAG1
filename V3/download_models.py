"""Télécharge les modèles V3 depuis HuggingFace Hub et les place sous src/models/.

Usage :
    python download_models.py            # télécharge les modèles manquants
    python download_models.py --check    # vérifie les fichiers locaux hors ligne
    python download_models.py --list     # affiche les chemins sans télécharger

Les modèles sont téléchargés une seule fois : si le dossier de destination
contient déjà les fichiers requis (config + poids), le téléchargement est
ignoré. La liste ci-dessous doit toujours correspondre aux modèles réellement
résolus par src/config.py (mêmes repo_id, mêmes clés d'environnement).
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from src.config import missing_model_files, _absolute_path, _is_local_model, settings, configured_models, model_status

BASE = Path(__file__).resolve().parent

# Doit rester synchronisé avec les modèles par défaut de src/config.py :
#   STT wolof   : M9and2M/whisper-small-wolof       (candidats locaux : src/models/stt_wolof_whisper_small,
#                                                     src/stt_wolof-whisper-small-lora, wolof-whisper-small-lora)
#   STT français: pierreguillou/whisper-medium-french (candidats locaux : src/models/stt_french_whisper_medium,
#                                                     src/stt_french_whisper_medium, whisper-medium-french)
#   WO→FR       : bilalfaye/nllb-200-distilled-600M-wo-fr-en
#   FR→WO       : Lahad/nllb200-francais-wolof
#   TTS         : soynade-research/Oolel-Voices
MODELES = [
    {
        "key": "STT_WO_MODEL_DIR",
        "repo_id": "M9and2M/whisper-small-wolof",
        "local_dir": BASE / "src" / "models" / "stt_wolof_whisper_small",
        "label": "STT WO — M9and2M/whisper-small-wolof",
    },
    {
        "key": "STT_FR_MODEL_DIR",
        "repo_id": "pierreguillou/whisper-medium-french",
        "local_dir": BASE / "src" / "models" / "stt_french_whisper_medium",
        "label": "STT FR — pierreguillou/whisper-medium-french",
    },
    {
        "key": "WO_FR_MODEL_DIR",
        "repo_id": "bilalfaye/nllb-200-distilled-600M-wo-fr-en",
        "local_dir": BASE / "src" / "models" / "wo_fr_nllb",
        "label": "WO→FR — bilalfaye/nllb-200-distilled-600M-wo-fr-en",
    },
    {
        "key": "FR_WO_MODEL_DIR",
        "repo_id": "Lahad/nllb200-francais-wolof",
        "local_dir": BASE / "src" / "models" / "fr_wo_nllb",
        "label": "FR→WO — Lahad/nllb200-francais-wolof",
    },
    {
        "key": "TTS_WO_MODEL_DIR",
        "repo_id": "soynade-research/Oolel-Voices",
        "local_dir": BASE / "src" / "models" / "tts_oolel_voices",
        "label": "TTS WO — soynade-research/Oolel-Voices",
    },
]


def already_downloaded(local_dir: Path, kind="transformers") -> bool:
    return not missing_model_files(local_dir, kind)


def _model_kind(model):
    if model["key"].startswith("STT_"):
        return "whisper"
    return "oolel" if model["key"] == "TTS_WO_MODEL_DIR" else "nllb"


def _destination(model):
    aliases = {
        "STT_WO_MODEL_DIR": ("STT_WO_MODEL_PATH", "STT_MODEL_PATH"),
        "STT_FR_MODEL_DIR": ("STT_FR_MODEL_PATH",),
        "WO_FR_MODEL_DIR": ("NLLB_WO_FR_MODEL_DIR", "NLLB_WO_FR_MODEL", "NLLB_WO_FR_MODEL_PATH"),
        "FR_WO_MODEL_DIR": ("NLLB_FR_WO_MODEL_DIR", "NLLB_FR_WO_MODEL", "NLLB_FR_WO_MODEL_PATH"),
        "TTS_WO_MODEL_DIR": ("OOLEL_MODEL_DIR", "OOLEL_TTS_REPO"),
    }
    for key in (model["key"], *aliases[model["key"]]):
        value = os.getenv(key, "").strip()
        if value:
            if key.endswith("_DIR") or _is_local_model(value):
                return _absolute_path(value)
            return model["local_dir"]
    return model["local_dir"]

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate local files without downloading")
    parser.add_argument("--list", action="store_true", help="Print supported model paths without downloading")
    args = parser.parse_args(argv)
    if args.check:
        invalid = False
        for name, (source, kind) in configured_models().items():
            status = model_status(source, kind)
            print(f"{name}: {status['state']} ({status['type']}) | {status['path']}")
            if status["missing"]:
                print("  Missing or invalid: " + ", ".join(status["missing"]))
            invalid = invalid or not status["files_available"]
        print("Required files checked offline; model loading, inference and API quotas are not tested.")
        print(f"LLM provider: {settings.LLM_PROVIDER}; configured: {settings.llm_ready}; remote availability: not tested.")
        if invalid:
            sys.exit(1)
        return
    print("\nChemins absolus à utiliser dans .env (aucun fichier .env modifié) :")
    for m in MODELES:
        print(f"  {m['key']}={_destination(m)}")
    print("Alias legacy : STT_MODEL_PATH/STT_WO_MODEL_PATH, STT_FR_MODEL_PATH, NLLB_WO_FR_MODEL, NLLB_FR_WO_MODEL, OOLEL_TTS_REPO.")
    print("Les clés *_MODEL_DIR sont prioritaires et désignent toujours des dossiers locaux.")
    if args.list:
        return
    print("=" * 60)
    print("TontumaBot V3 — Téléchargement des modèles")
    print("=" * 60)

    for m in MODELES:
        source = getattr(settings, m["key"])
        kind = _model_kind(m)
        status = model_status(source, kind)
        if status["files_available"]:
            print(f"Already available: {m['label']} | {status['path']}")
            continue
        if not _is_local_model(source) and source != m["repo_id"]:
            raise ValueError(f"Configured source {source} differs from {m['repo_id']}; refusing to download unrelated weights.")
        local_dir = _absolute_path(source) if _is_local_model(source) else _destination(m)
        local_dir.mkdir(parents=True, exist_ok=True)

        if already_downloaded(local_dir, kind):
            print(f"\n✅ Déjà présent : {m['label']}")
            print(f"   Dossier      : {local_dir}")
            continue

        print(f"\n⬇️  Téléchargement : {m['label']}")
        print(f"   Hub          : {m['repo_id']}")
        print(f"   Destination  : {local_dir}")

        try:
            snapshot_download(
                repo_id=m["repo_id"],
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=[
                    "*.msgpack",
                    "flax_model*",
                    "tf_model*",
                    "rust_model*",
                ],
            )
            missing = missing_model_files(local_dir, kind)
            if missing:
                raise RuntimeError(f"Incomplete download in {local_dir}: {', '.join(missing)}")
            print("   ✅ Téléchargé avec succès.")
        except Exception as e:
            print(f"   ❌ Échec : {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Tous les modèles sont prêts.")
    print("=" * 60)

    print("\nChemins absolus à utiliser dans .env (aucun fichier .env modifié) :")
    for m in MODELES:
        print(f"  {m['key']}={_destination(m)}")
    print("Alias legacy : STT_MODEL_PATH/STT_WO_MODEL_PATH, STT_FR_MODEL_PATH, NLLB_WO_FR_MODEL, NLLB_FR_WO_MODEL, OOLEL_TTS_REPO.")
    print("Les clés *_MODEL_DIR sont prioritaires et désignent toujours des dossiers locaux.")

if __name__ == "__main__":
    main()
