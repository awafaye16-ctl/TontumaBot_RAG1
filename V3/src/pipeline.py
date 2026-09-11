import asyncio
import re
import time
from typing import Optional

import vectorstore
from config import settings
from language.detector import detect_language
from translation.nllb import wolof_to_french, french_to_wolof
from intent.router import detect_intent
from retrieval.reranker import rerank
from generation.llm import generate, generate_stream
from memory.conversation import memoire


_GREETING_RESPONSE = (
    "Bonjour ! Je suis TontumaBot, votre assistant administratif sénégalais. "
    "Posez-moi une question sur les démarches, horaires ou localisation des services."
)

_GREETING_KEYWORDS = [
    "bonjour", "salut", "salam", "hello", "hi", "coucou",
    "bonsoir", "comment ça va", "ça va", "ca va",
    "jerejef", "waaw", "na nga def", "naka nga def",
    "merci", "jërëjëf", "ak jërëjëf", "jërejëf", "jerejef lool",
]


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"^\s*[-*+•]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def reformuler_question(question_actuelle: str, historique: list[dict]) -> Optional[str]:
    """Reformule une question de suivi en question autonome et en français, via le LLM.

    Si l'historique est vide, ou si la reformulation échoue/ne produit rien
    d'exploitable, retourne None : l'appelant doit alors utiliser la question
    d'origine (et, si elle est en wolof, la faire traduire normalement — ne
    jamais renvoyer le texte brut potentiellement wolof ici, sous peine de
    court-circuiter la traduction WO->FR dans _prepare()).
    Utilise les 4 derniers messages pour le contexte de reformulation.
    """
    if not historique:
        return None
    recent = historique[-4:]
    dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    prompt = (
        "Tu es un assistant de reformulation. L'historique et la question actuelle "
        "sont en français. Reformule la question actuelle en une question autonome "
        "et complète en français, compréhensible sans le contexte de la conversation. "
        "Réponds UNIQUEMENT en français, même si la question actuelle est formulée "
        "dans une autre langue (ex. wolof) : traduis-la et reformule-la en français. "
        "Réponds uniquement avec la question reformulée, rien d'autre.\n\n"
        f"Historique:\n{dialogue}\n\n"
        f"Question actuelle: {question_actuelle}\n\n"
        "Question reformulée (en français):"
    )
    try:
        from generation.llm import _groq_generate
        reformulee = _groq_generate(prompt, "", plain=True).strip()
        return reformulee or None
    except Exception:
        return None


def normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    aliases = {"fr": "fr", "french": "fr", "wo": "wo", "wolof": "wo", "auto": None, "": None}
    value = language.strip().lower()
    if value not in aliases:
        raise ValueError("Langue non supportée : choisir fr, wo ou auto")
    return aliases[value]


def _is_greeting(text: str) -> bool:
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    normalized = " ".join(normalized.split())
    return normalized in _GREETING_KEYWORDS


def _common_overlap(a: str, b: str, max_len: int = 220) -> int:
    """Longueur du plus grand suffixe de `a` qui est aussi préfixe de `b`.

    Sert à ne pas dupliquer le chevauchement (CHUNK_OVERLAP) déjà présent
    entre deux chunks contigus lors de leur recollage.
    """
    max_len = min(max_len, len(a), len(b))
    for length in range(max_len, 0, -1):
        if a[-length:] == b[:length]:
            return length
    return 0


def _stitch_contiguous(texts_by_index: dict) -> str:
    """Recolle des chunks contigus (par chunk_index croissant) en un texte continu."""
    ordered = sorted(texts_by_index.items())
    result = ""
    prev_idx = None
    for idx, text in ordered:
        if not result:
            result = text
        elif prev_idx is not None and idx == prev_idx + 1:
            overlap = _common_overlap(result, text)
            result += text[overlap:]
        else:
            result += "\n[…]\n" + text
        prev_idx = idx
    return result


def _expand_context(chunks: list[dict]) -> str:
    """Construit le contexte envoyé au LLM en complétant chaque passage retenu
    par ses chunks voisins immédiats (même document_id, chunk_index ± 1).

    Un chunk seul peut couper une procédure au milieu (ex. étapes 2 à 5 d'une
    liste numérotée absentes du passage retenu). Étendre aux voisins directs
    évite de renvoyer une réponse partielle alors que l'information complète
    est présente dans le document source.
    """
    if not chunks:
        return ""
    doc_cache: dict[str, dict] = {}
    by_doc: dict[str, dict[int, str]] = {}
    standalone = []
    for chunk in chunks:
        document_id = chunk.get("document_id")
        idx = chunk.get("metadata", {}).get("chunk_index")
        if not document_id or not isinstance(idx, int):
            standalone.append(chunk)
            continue
        if document_id not in doc_cache:
            doc_cache[document_id] = vectorstore.get_document_chunks(document_id)
        doc_chunks = doc_cache[document_id]
        bucket = by_doc.setdefault(document_id, {})
        for neighbor_idx in (idx - 1, idx, idx + 1):
            entry = doc_chunks.get(neighbor_idx)
            if entry is not None:
                bucket[neighbor_idx] = entry[1]
    parts = [f"[Passage {chunk['rank']} | source: {chunk['source']} | id: {chunk['id']}]\n{chunk['text']}"
             for chunk in standalone]
    seen_docs: list[str] = []
    titles = {chunk.get("document_id"): chunk.get("title", "") for chunk in chunks if chunk.get("document_id")}
    for chunk in chunks:
        document_id = chunk.get("document_id")
        if document_id and document_id in by_doc and document_id not in seen_docs:
            seen_docs.append(document_id)
    for document_id in seen_docs:
        text = _stitch_contiguous(by_doc[document_id])
        parts.append(f"[Document : {titles.get(document_id) or document_id}]\n{text}")
    return "\n\n".join(parts)


def _rag_pipeline(question_fr: str, intent: str, fetch_k: int = 20, mmr_k: int = 8, top_k: int = None,
                   alpha: float = 0.4) -> tuple[str, dict]:
    started = time.perf_counter()
    top_k = settings.RERANKER_TOP_K if top_k is None else top_k
    t0 = time.perf_counter()
    candidates = vectorstore.hybrid_search(question_fr, k=fetch_k, alpha=alpha)
    t_hybrid = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    selected = vectorstore.mmr_search(question_fr, k=mmr_k, fetch_k=fetch_k, candidates=candidates) if candidates else []
    t_mmr = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    reranked = rerank(question_fr, [item[1] for item in selected], top_k=top_k) if selected else []
    t_rerank = round((time.perf_counter() - t0) * 1000, 1)
    threshold = getattr(settings, "RERANKER_MIN_SCORE", None)
    accepted = [(idx, score, text) for idx, score, text in reranked if threshold is None or score >= threshold]
    chunks = []
    for rank, (idx, score, text) in enumerate(accepted, 1):
        cid, _, _, metadata = selected[idx]
        chunks.append({
            "rank": rank, "id": cid, "score": round(score, 4), "text": text,
            "source": metadata.get("source", ""), "title": metadata.get("title", ""),
            "document_id": metadata.get("document_id", ""), "metadata": metadata,
        })
    context = _expand_context(chunks)
    return context, {
        "intent": intent, "n_candidates": len(candidates), "n_mmr": len(selected), "alpha": alpha,
        "n_reranked": len(reranked), "n_filtered": len(chunks), "min_score": threshold,
        "latency_hybrid_ms": t_hybrid, "latency_mmr_ms": t_mmr, "latency_rerank_ms": t_rerank,
        "latency_total_ms": round((time.perf_counter() - started) * 1000, 1),
        "reranker_scores": [round(score, 4) for _, score, _ in reranked], "chunks": chunks,
    }


def _rag_seed(question_fr: str, intent: str, seed_docs: list[str], seed_filtered: Optional[list[dict]]) -> tuple[str, dict]:
    from retrieval.hybrid import HybridRetriever
    from retrieval.filtered import FilteredSearch

    started = time.perf_counter()
    if intent == "orientation" and seed_filtered:
        texts = [item["text"] for item in FilteredSearch(seed_filtered).search(question_fr, k=8)]
        search_type = "filtered (seed)"
    else:
        ranked = HybridRetriever(seed_docs).search(question_fr, k=8) if seed_docs else []
        texts = [seed_docs[idx] for idx, _ in ranked]
        search_type = "BM25 (seed)"
    texts = list(dict.fromkeys(texts))
    reranked = rerank(question_fr, texts, top_k=settings.RERANKER_TOP_K) if texts else []
    chunks = [{"rank": rank, "id": f"seed-{idx}", "source": "seed", "score": round(score, 4), "text": text}
              for rank, (idx, score, text) in enumerate(reranked, 1)]
    return "\n\n".join(chunk["text"] for chunk in chunks), {
        "search_type": search_type, "intent": intent, "n_candidates": len(texts),
        "n_reranked": len(reranked), "n_filtered": len(chunks), "chunks": chunks,
        "reranker_scores": [round(score, 4) for _, score, _ in reranked],
        "latency_total_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _prepare(unified_text, provider, tts, tts_engine, seed_docs, seed_filtered, input_lang,
             question_recherche=None, historique=None):
    if not isinstance(unified_text, str) or not unified_text.strip():
        raise ValueError("Question vide")
    if len(unified_text) > 12000:
        raise ValueError("Question trop longue (maximum 12000 caractères)")
    if provider not in ("groq", "gemini", "local"):
        raise ValueError("Fournisseur LLM non supporté")
    if tts_engine not in (None, "oolel", "speecht5", "edge"):
        raise ValueError("Moteur TTS non supporté")
    language = normalize_language(input_lang)
    lang = language or detect_language(unified_text)
    trace = {
        "input_text": unified_text, "input_type": "text", "provider": provider,
        "input_lang": lang, "lang_source": "provided" if language else "auto",
        "tts_requested": tts, "tts_engine_requested": tts_engine,
    }
    if question_recherche:
        trace["question_reformulee"] = question_recherche
    if _is_greeting(unified_text):
        trace.update({"is_greeting": True, "context": "", "retrieval": {"source": "salutation", "n_reranked": 0, "chunks": []}})
        return unified_text, "", trace
    question_fr = (question_recherche or unified_text).strip()
    if lang == "wo" and not question_recherche:
        t0 = time.perf_counter()
        question_fr, _ = wolof_to_french(question_fr)
        if not question_fr.strip():
            raise RuntimeError("La traduction WO vers FR a retourné un texte vide")
        trace["wolof_to_french"] = {"model": settings.NLLB_WO_FR_MODEL, "result": question_fr,
                                      "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    intent = detect_intent(question_fr)
    count = vectorstore.count()
    if count:
        # Une question traduite du wolof peut employer un vocabulaire différent de
        # celui des documents (ex. « certificat de mort » traduit pour « acte de
        # décès ») : la recherche lexicale (BM25) perd alors tout signal. On
        # privilégie davantage la similarité sémantique (vectorielle), plus
        # robuste aux reformulations/synonymes introduits par la traduction.
        alpha = 0.2 if lang == "wo" else 0.4
        context, retrieval = _rag_pipeline(question_fr, intent, alpha=alpha)
        retrieval["source"] = "chromadb"
    elif seed_docs:
        context, retrieval = _rag_seed(question_fr, intent, seed_docs, seed_filtered)
        retrieval["source"] = "seed"
    else:
        context, retrieval = "", {"source": "aucun document", "n_candidates": 0, "n_reranked": 0, "chunks": []}
    trace.update({"intent": intent, "context": context, "retrieval": retrieval, "n_docs_in_db": count, "n_chunks_in_db": count})
    return question_fr, context, trace


def _finish(response_fr, trace, tts, tts_engine, tts_out, started):
    if not isinstance(response_fr, str) or not response_fr.strip():
        raise RuntimeError("La génération LLM a retourné une réponse vide")
    response_fr = response_fr.strip()
    result = {"trace": trace, "response_fr": response_fr, "response": response_fr,
              "sources": [{key: chunk[key] for key in ("id", "source", "title", "document_id", "score") if key in chunk}
                          for chunk in trace["retrieval"].get("chunks", [])]}
    lang = trace["input_lang"]
    if lang == "wo":
        t0 = time.perf_counter()
        translated, _ = french_to_wolof(_strip_markdown(response_fr))
        if not translated.strip():
            raise RuntimeError("La traduction FR vers WO a retourné un texte vide")
        result.update({"response_wo": translated, "response": translated})
        trace["french_to_wolof"] = {"model": settings.NLLB_FR_WO_MODEL, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if tts:
        t0 = time.perf_counter()
        try:
            from tts_Ooleil.tts import synthesize_with_metadata
            audio = synthesize_with_metadata(_strip_markdown(result["response"]), tts_out, engine=tts_engine, language=lang)
            result["audio"] = audio["path"]
            trace["tts"] = {key: value for key, value in audio.items() if key != "path"}
            trace["tts"]["status"] = "ok"
        except Exception as exc:
            trace["tts"] = {"status": "error", "error_type": type(exc).__name__}
            result["warnings"] = ["La réponse texte est disponible, mais la synthèse vocale a échoué."]
        trace["tts"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


def answer(unified_text: str, provider: str = "groq", tts: bool = False, tts_engine: Optional[str] = None,
           tts_out: str = "response.wav", seed_docs: Optional[list[str]] = None,
           seed_filtered: Optional[list[dict]] = None, input_lang: Optional[str] = None,
           session_id: Optional[str] = None) -> dict:
    started = time.perf_counter()
    historique = memoire.historique_pour_question(session_id) if session_id else []
    question_recherche = reformuler_question(unified_text, historique) if historique else None
    question, context, trace = _prepare(unified_text, provider, tts, tts_engine, seed_docs, seed_filtered,
                                        input_lang, question_recherche=question_recherche,
                                        historique=historique)
    t0 = time.perf_counter()
    response_fr = _GREETING_RESPONSE if trace.get("is_greeting") else generate(question, context, provider=provider, plain=trace["input_lang"] == "wo")
    trace["llm"] = {"provider": provider, "called": bool(context) and not trace.get("is_greeting", False),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    result = _finish(response_fr, trace, tts, tts_engine, tts_out, started)
    if session_id:
        # Toujours stocker le français en mémoire (jamais le wolof brut/traduit) :
        # la reformulation de suivi doit pouvoir s'appuyer sur un historique
        # cohérent, faute de quoi la traduction WO->FR est court-circuitée à la
        # question suivante (question_recherche devient non vide) et le RAG reçoit
        # du texte wolof au lieu du français attendu.
        result["session"] = memoire.ajouter_tour(session_id, question, result.get("response_fr", response_fr))
    return result


async def answer_stream(unified_text: str, provider: str = "groq", tts: bool = False, tts_engine: Optional[str] = None,
                        tts_out: str = "response.wav", seed_docs: Optional[list[str]] = None,
                        seed_filtered: Optional[list[dict]] = None, input_lang: Optional[str] = None,
                        session_id: Optional[str] = None):
    started = time.perf_counter()
    historique = memoire.historique_pour_question(session_id) if session_id else []
    question_recherche = reformuler_question(unified_text, historique) if historique else None
    question, context, trace = await asyncio.to_thread(_prepare, unified_text, provider, tts, tts_engine,
                                                       seed_docs, seed_filtered, input_lang,
                                                       question_recherche=question_recherche,
                                                       historique=historique)
    yield {"type": "lang", "lang": trace["input_lang"]}
    if "wolof_to_french" in trace:
        yield {"type": "query_fr", "text": question}
    if "question_reformulee" in trace:
        yield {"type": "query_reformulee", "text": trace["question_reformulee"]}
    yield {"type": "context_ready", "n_chunks": len(trace["retrieval"].get("chunks", []))}
    t0 = time.perf_counter()
    parts = []
    if trace.get("is_greeting"):
        parts.append(_GREETING_RESPONSE)
        yield {"type": "token", "text": _GREETING_RESPONSE}
    else:
        async for token in generate_stream(question, context, provider=provider, plain=trace["input_lang"] == "wo"):
            parts.append(token)
            yield {"type": "token", "text": token}
    trace["llm"] = {"provider": provider, "called": bool(context) and not trace.get("is_greeting", False),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    result = await asyncio.to_thread(_finish, "".join(parts), trace, tts, tts_engine, tts_out, started)
    if session_id:
        # Voir le commentaire équivalent dans answer() : toujours stocker le
        # français en mémoire, jamais le wolof brut.
        result["session"] = memoire.ajouter_tour(session_id, question, result.get("response_fr", "".join(parts)))
    yield {"type": "done", **result}
