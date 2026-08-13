from retrieval.hybrid import HybridRetriever
from retrieval.filtered import FilteredSearch

DOCUMENTS = [
    "Pour obtenir une carte d'identité nationale, présentez une copie de l'extrait de naissance et deux photos au guichet de la mairie.",
    "Créer une entreprise au Sénégal nécessite la carte d'identité, un extrait du registre de commerce et un justificatif de domicile auprès de l'APIX.",
    "L'extrait de naissance se demande à la mairie de votre lieu de naissance avec le carnet de famille.",
    "Pour un passeport biométrique, déposez le formulaire et l'extrait de naissance au commissariat.",
]


def test_bm25_retourne_resultats():
    h = HybridRetriever(DOCUMENTS)
    results = h.search("créer entreprise")
    assert results, "La recherche BM25 devrait retourner des résultats"
    idx, score = results[0]
    assert idx == 1, "Le document le plus pertinent devrait être l'index 1 (entreprise)"


def test_bm25_pertinence_extrait_naissance():
    h = HybridRetriever(DOCUMENTS)
    results = dict(h.search("extrait naissance", k=5))
    assert 2 in results, "Le doc 2 (extrait de naissance) devrait être retrouvé"
    assert 0 in results, "Le doc 0 (extrait de naissance) devrait être retrouvé"


def test_bm25_retourne_vide_sans_terme_commun():
    h = HybridRetriever(DOCUMENTS)
    results = h.search("xyzabc inconnu mot")
    assert results == []


def test_filtered_search_par_lieu():
    docs = [
        {"text": "La demande se fait à la mairie.", "meta": {"lieu": True}},
        {"text": "L'APIX gère la création d'entreprise.", "meta": {"service": True}},
    ]
    f = FilteredSearch(docs)
    hits = f.search("Où dois-je déposer ma demande ?")
    assert hits and hits[0]["meta"].get("lieu")


def test_filtered_search_par_service():
    docs = [
        {"text": "La demande se fait à la mairie.", "meta": {"lieu": True}},
        {"text": "L'APIX gère la création d'entreprise.", "meta": {"service": True}},
    ]
    f = FilteredSearch(docs)
    hits = f.search("Quel service s'occupe de la création d'entreprise ?")
    assert hits and hits[0]["meta"].get("service")


def test_hybrid_sans_embedder_retombe_sur_bm25():
    h = HybridRetriever(DOCUMENTS, embedder=None)
    results = h.search("créer entreprise")
    assert results, "Sans embedder, BM25 seul doit suffire"