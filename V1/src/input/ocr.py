"""OCR : extrait le texte d'une image ou d'un document (pytesseract)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

from PIL import Image
import pytesseract

OCR_LANGS = settings.OCR_LANGS or "fra+eng"


def ocr_image(image_path: str, lang: str = None) -> str:
    """
    Extrait le texte présent dans l'image/document.
    `lang` : combinaison tesseract (ex: 'fra+eng'). Par défaut depuis .env.
    """
    lang = lang or OCR_LANGS
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang=lang).strip()


def read(image_path: str) -> str:
    """Point d'entrée : image -> texte."""
    return ocr_image(image_path)