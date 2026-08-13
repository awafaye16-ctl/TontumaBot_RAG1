"""Router d'intention : détermine si la requête FR relève d'une
procédure (recherche hybride) ou d'une orientation (recherche filtrée).
"""
import re

ORIENTATION_KEYWORDS = {
    "où", "aller", "déposer", "rendre", "adresse", "bureau", "guichet",
    "service", "administration", "contacter", "adresser", "localiser",
    "quel service", "quelle administration", "quel ministère", "commissariat",
    "mairie", "lieu", "endroit", "situé", "située", "combien", "coûte",
    "coûtent", "gratuit", "coût", "tarif", "prix", "frais","batiment", "immeuble", "centre", "agence", "poste", "ambassade",
}
# Mots courts qui doivent matcher sur les frontières de mot uniquement.
_WORD_ONLY = {"où", "aller", "rendre", "mairie", "lieu", "prix", "coût", "frais", "combien"}


def _contains_keyword(q: str, kw: str) -> bool:
    if kw in _WORD_ONLY:
        return re.search(rf"\b{re.escape(kw)}\b", q) is not None
    return kw in q


def detect_intent(query_fr: str) -> str:
    """
    Retourne 'procedure' ou 'orientation'.
    Les mots-clés d'orientation (lieu, service, coût) priment.
    """
    q = query_fr.lower()
    if any(_contains_keyword(q, kw) for kw in ORIENTATION_KEYWORDS):
        return "orientation"
    return "procedure"


if __name__ == "__main__":
    tests = [
        "Quels sont les documents nécessaires pour une carte d'identité ?",
        "Où dois-je déposer cette demande ?",
        "Quel service s'occupe de cette procédure ?",
        "Comment créer une entreprise au Sénégal ?",
    ]
    for t in tests:
        print(f"{detect_intent(t):10} | {t}")