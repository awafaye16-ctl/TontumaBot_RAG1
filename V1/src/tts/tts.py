"""TTS : réponse vocale wolof (edge-tts)."""
import asyncio
import edge_tts

# Voix wolof : edge-tts ne la propose pas nativement pour l'instant ;
# on utilise une voix FR par défaut. Remplacer par une voix wolof si disponible.
VOICE = "fr-FR-DeniseNeural"


async def _synth(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(out_path)


def synthesize(text_wolof: str, out_path: str) -> str:
    asyncio.run(_synth(text_wolof, out_path))
    return out_path