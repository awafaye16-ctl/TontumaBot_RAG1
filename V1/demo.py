"""Démo CLI : valide le flux du pipeline sans clé API ni modèles lourds.

- Détection de langue
- Router d'intention
- Recherche hybride / filtrée
- Génération simulée (si pas de clé API) : remplacée par le contexte retrouvé.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from language.detector import detect_language
from intent.router import detect_intent
from retrieval.hybrid import HybridRetriever
from retrieval.filtered import FilteredSearch

DOCUMENTS = [
    "Pour obtenir une carte d'identité nationale, présentez une copie de l'extrait de naissance, deux photos d'identité et le reçu de paiement au guichet de la mairie.",
    "Créer une entreprise au Sénégal nécessite la carte d'identité, un extrait du registre de commerce et un justificatif de domicile auprès de l'APIX.",
    "L'extrait de naissance se demande à la mairie de votre lieu de naissance avec le carnet de famille et la pièce d'identité du parent.",
    "Pour un passeport biométrique, déposez le formulaire et l'extrait de naissance au commissariat ou à la direction de la police nationale.",
]

FILTERED_DOCS = [
    {"text": "La demande d'extrait de naissance se fait à la mairie de votre lieu de naissance, service de l'état civil.",
     "meta": {"lieu": True}},
    {"text": "L'APIX gère les formalités de création d'entreprise. Contactez le guichet unique au boulevard de Dakar.",
     "meta": {"service": True, "lieu": True}},
    {"text": "Le passeport se retire à la direction de la police nationale, département des passeports.",
     "meta": {"service": True, "lieu": True}},
]


def run_question(q: str, hybrid: HybridRetriever, filtered: FilteredSearch):
    print(f"\n{'=' * 62}\nQuestion : {q}")
    print(f"Langue détectée : {detect_language(q).upper()}")

    intent = detect_intent(q)
    print(f"Intention : {intent}")

    if intent == "orientation":
        hits = filtered.search(q)
        print("Recherche filtrée :")
        for h in hits:
            print(f"  - {h['text'][:90]}...")
    else:
        ranked = hybrid.search(q)
        print("Recherche hybride (BM25) :")
        for i, s in ranked[:3]:
            print(f"  [{s:6.3f}] {DOCUMENTS[i][:90]}...")


def main():
    hybrid = HybridRetriever(DOCUMENTS)
    filtered = FilteredSearch(FILTERED_DOCS)

    questions = [
        "dama beug wout kayitu juddu?",
        "Où dois-je déposer ma demande de carte d'identité ?",
        "Comment créer une entreprise au Sénégal ?",
    ]
    for q in questions:
        run_question(q, hybrid, filtered)


if __name__ == "__main__":
    main()