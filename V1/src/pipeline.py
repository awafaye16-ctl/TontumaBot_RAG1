"""Pipeline principal TontumaBot V1.

Flux :
entrée (texte/audio/image) -> texte unifié -> détection langue
-> (wolof ? NLLB WO->FR) -> router d'intention
-> recherche (hybride vectorielle BM25+embeddings ou filtrée) -> contexte FR
-> LLM -> réponse -> (wolof ? NLLB FR->WO) -> réponse texte (+ TTS optionnel)

La recherche s'appuie sur la base vectorielle ChromaDB si elle contient des
documents, sinon sur une base de démonstration (seed).
"""
import vectorstore
from language.detector import detect_language
from translation.nllb import wolof_to_french, french_to_wolof
from intent.router import detect_intent
from retrieval.filtered import FilteredSearch
from generation.llm import generate


def _search_context(question_fr: str, intent: str) -> tuple[str, str]:
    """Retourne (contexte_fr, type_recherche)."""
    if intent == "orientation":
        hits = vectorstore.vector_search(question_fr, k=3)
        if not hits:
            return "", "vectoriel"
        context = "\n\n".join(doc for _, doc, _, _ in hits)
        return context, "vectoriel (orientation)"

    hits = vectorstore.hybrid_search(question_fr, k=5)
    if not hits:
        return "", "hybride"
    context = "\n\n".join(doc for _, doc, _, _ in hits)
    return context, "hybride (BM25 + vectoriel)"


def answer(
    unified_text: str,
    provider: str = "groq",
    tts: bool = False,
    tts_out: str = "response.mp3",
    seed_docs: list[str] = None,
    seed_filtered: list[dict] = None,
    force_seed: bool = False,
) -> dict:
    """Traite une question unifiée (texte) et retourne la réponse complète."""
    lang = detect_language(unified_text)
    trace = {"input_lang": lang}

    question_fr = unified_text
    if lang == "wo":
        question_fr, d1 = wolof_to_french(unified_text)
        trace["wolof_to_french"] = {"query": question_fr, "seconds": d1}

    intent = detect_intent(question_fr)
    trace["intent"] = intent

    n_docs = vectorstore.count() if not force_seed else 0
    if n_docs > 0:
        context_fr, search_type = _search_context(question_fr, intent)
    elif seed_docs:
        # Fallback : base de démonstration en mémoire
        hybrid = _HybridSeed(seed_docs)
        filtered = FilteredSearch(seed_filtered or [])
        if intent == "orientation":
            hits = filtered.search(question_fr)
            context_fr = "\n\n".join(h["text"] for h in hits)
            search_type = "filtrée (seed)"
        else:
            ranked = hybrid.search(question_fr)
            context_fr = "\n\n".join(hybrid.documents[i] for i, _ in ranked)
            search_type = "hybride (seed)"
    else:
        context_fr = ""
        search_type = "aucun document"

    trace["search"] = search_type
    trace["context"] = context_fr
    trace["n_docs"] = n_docs

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
        synthesize(text_for_tts, tts_out)
        response["audio"] = tts_out
        trace["tts"] = tts_source()

    return response


class _HybridSeed:
    """Mini recherche hybride BM25 en mémoire (pour la base seed)."""

    def __init__(self, documents: list[str]):
        from retrieval.hybrid import HybridRetriever

        self._impl = HybridRetriever(documents)
        self.documents = documents

    def search(self, query: str):
        return self._impl.search(query)


# Compatibilité avec l'ancienne signature (tests)
def build_retrievers(documents: list[str], filtered_docs: list[dict]):
    """Conservé pour compatibilité : retourne deux objets de recherche seed."""
    hybrid = _HybridSeed(documents)
    filtered = FilteredSearch(filtered_docs)
    return hybrid, filtered


def answer_legacy(unified_text, hybrid, filtered, provider="groq", tts=False, tts_out="response.mp3"):
    """Ancienne signature : délègue à `answer` en forçant la base seed."""
    return answer(
        unified_text,
        provider=provider,
        tts=tts,
        tts_out=tts_out,
        seed_docs=hybrid.documents,
        seed_filtered=filtered.documents,
        force_seed=True,
    )