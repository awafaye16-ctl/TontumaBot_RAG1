import re

WOOF_WORDS = {
    "ana", "naka", "lu", "lan", "fan", "ñu", "woy", "jëf", "jëm", "ci", "la",
    "dama", "danga", "dafa", "ñu", "man", "yow", "muy", "bu", "su", "te", "tey",
    "bés", "bam", "kayitu", "juddu", "jërejëf", "dimbal", "soxla", "doctoor",
    "metit", "beug", "wout", "fu", "loo", "déef", "ngir", "nelaw", "tontu",
    "xiif", "naari", "benen", "jigéen", "góor", "xale", "mag", "ñu", "bii",
}


def detect_language(text: str) -> str:
    """
    Détecte si `text` est du wolof ou du français.
    Retourne 'wo' ou 'fr'.
    """
    words = set(re.findall(r"[\wàâäéèêëîïôöùûüç]+", text.lower()))
    if not words:
        return "fr"
    wolof_hits = len(words & WOOF_WORDS)
    return "wo" if wolof_hits >= 1 else "fr"


if __name__ == "__main__":
    tests = [
        "dama beug wout kayitu juddu?",
        "Comment puis-je obtenir un extrait de naissance ?",
        "Fan lañuy def demande bi ?",
        "Quels sont les documents nécessaires pour un passeport ?",
    ]
    for t in tests:
        print(f"{detect_language(t).upper():2} | {t}")