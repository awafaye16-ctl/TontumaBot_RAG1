import re

WOLOF_WORDS = {
    "ana", "naka", "lu", "lan", "fan", "ñu", "woy", "jëf", "jëm", "ci", "la",
    "dama", "danga", "dafa", "ñu", "man", "yow", "muy", "bu", "su", "te", "tey",
    "bés", "bam", "kayitu", "juddu", "jërejëf", "dimbal", "soxla", "doctoor",
    "metit", "beug", "wout", "fu", "loo", "déef", "ngir", "nelaw", "tontu",
    "xiif", "naari", "benen", "jigéen", "góor", "xale", "mag", "ñu", "bii",
}


def detect_language(text: str) -> str:
    words = set(re.findall(r"[\wàâäéèêëîïôöùûüç]+", text.lower()))
    if not words:
        return "fr"
    wolof_hits = len(words & WOLOF_WORDS)
    return "wo" if wolof_hits >= 1 else "fr"
