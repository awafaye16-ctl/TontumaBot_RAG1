"""Pipeline V2 — architecture complète avec Reranker.

Flux :
  entrée (texte/audio) -> texte -> détection langue
  -> (wolof ? NLLB WO->FR) -> router d'intention
  -> Retriever (hybride BM25 + vectoriel OU filtered)
  -> Reranker (cross-encoder)
  -> Contexte FR -> LLM -> réponse FR
  -> (wolof ? NLLB FR->WO) -> TTS optionnel
"""
import vectorstore
from language.detector import detect_language
from translation.nllb import wolof_to_french, french_to_wolof
from intent.router import detect_intent
from retrieval.filtered import FilteredSearch
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import rerank
from generation.llm import generate


def _search_and_rerank(question_fr: str, intent: str, seed_docs=None, seed_filtered=None):
    """Retourne (contexte_fr, type_recherche, rerank_info)."""
    n_docs = vectorstore.count()

    # 1. Chercher dans ChromaDB si des documents sont indexés
    if n_docs > 0:
        hits = vectorstore.hybrid_search(question_fr, k=5)
        texts = [doc for _, doc, _, _ in hits]
        search_type = "hybride BM25+vectoriel"
    # 2. Sinon, utiliser les seed docs en mémoire
    elif seed_docs:
        # Pour l'intention orientation, utiliser filtered search
        if intent == "orientation" and seed_filtered:
            fs = FilteredSearch(seed_filtered)
            filtered = fs.search(question_fr, k=3)
            if filtered:
                texts = [d["text"] for d in filtered]
                search_type = "filtered (orientation)"
            else:
                texts = seed_docs
                search_type = "seed (pas de match filtered)"
        else:
            hybrid = HybridRetriever(seed_docs)
            ranked = hybrid.search(question_fr, k=5)
            texts = [seed_docs[i] for i, _ in ranked] if ranked else seed_docs[:3]
            search_type = "hybride (seed)"
    else:
        return "", "aucun document", {}

    if not texts:
        return "", search_type, {}

    # Reranker
    reranked = rerank(question_fr, texts)
    context = "\n\n".join(doc for _, _, doc in reranked)
    rerank_info = {
        "input_docs": len(texts),
        "output_docs": len(reranked),
        "scores": [round(s, 4) for _, s, _ in reranked],
    }
    return context, search_type, rerank_info


def answer(
    unified_text: str,
    provider: str = "groq",
    tts: bool = False,
    tts_engine: str = None,
    tts_out: str = "response.mp3",
    seed_docs: list[str] = None,
    seed_filtered: list[dict] = None,
) -> dict:
    lang = detect_language(unified_text)
    trace = {"input_lang": lang}

    question_fr = unified_text
    if lang == "wo":
        question_fr, d1 = wolof_to_french(unified_text)
        trace["wolof_to_french"] = {"query": question_fr, "seconds": d1}

    intent = detect_intent(question_fr)
    trace["intent"] = intent

    context_fr, search_type, rerank_info = _search_and_rerank(
        question_fr, intent, seed_docs, seed_filtered
    )
    trace["search"] = search_type
    trace["reranker"] = rerank_info
    trace["context"] = context_fr

    response_fr = generate(question_fr, context_fr, provider=provider)
    response = {"trace": trace, "response_fr": response_fr, "response": response_fr}

    if lang == "wo":
        response_wo, d2 = french_to_wolof(response_fr)
        response["response_wo"] = response_wo
        response["response"] = response_wo
        trace["french_to_wolof"] = {"seconds": d2}

    if tts:
        from tts.tts import synthesize, source as tts_source
        text_for_tts = response.get("response_wo", response_fr)
        synthesize(text_for_tts, tts_out, engine=tts_engine)
        response["audio"] = tts_out
        trace["tts"] = tts_source(tts_engine)

    return response
