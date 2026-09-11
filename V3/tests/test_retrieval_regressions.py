import importlib.util
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch


SRC = Path(__file__).resolve().parents[1] / "src"
SETTINGS = types.SimpleNamespace(
    EMBED_MODEL="test-embedder", BASE_DIR="unused-test-database",
    RERANKER_MODEL="test-reranker", RERANKER_TOP_K=3,
)


class NotFoundError(Exception):
    pass


class InvalidCollectionException(Exception):
    pass


class FakeBM25:
    def __init__(self, documents):
        self.documents = documents

    def get_scores(self, query):
        return [sum(document.count(word) for word in query) for document in self.documents]


CHROMA = types.SimpleNamespace(
    PersistentClient=Mock(side_effect=AssertionError("Real database access is forbidden")),
    errors=types.SimpleNamespace(
        NotFoundError=NotFoundError, InvalidCollectionException=InvalidCollectionException,
    ),
)
NUMPY = types.SimpleNamespace(
    ndarray=list, array=lambda value: value,
    dot=lambda left, right: sum(a * b for a, b in zip(left, right)),
)


def load_module(name, relative_path, dependencies=None):
    spec = importlib.util.spec_from_file_location(name, SRC / relative_path)
    module = importlib.util.module_from_spec(spec)
    saved_path = list(sys.path)
    try:
        with patch.dict(sys.modules, {name: module, **(dependencies or {})}):
            spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
    return module


vectorstore = load_module("storage_under_test", "vectorstore.py", {
    "config": types.SimpleNamespace(settings=SETTINGS),
    "chromadb": CHROMA, "numpy": NUMPY,
})
ingestion = load_module("ingestion_under_test", "ingestion.py", {"vectorstore": vectorstore})
hybrid = load_module("hybrid_under_test", "retrieval/hybrid.py", {
    "rank_bm25": types.SimpleNamespace(BM25Okapi=FakeBM25),
})
reranker = load_module("reranker_under_test", "retrieval/reranker.py", {
    "config": types.SimpleNamespace(settings=SETTINGS),
})
router = load_module("router_under_test", "intent/router.py")
detector = load_module("detector_under_test", "language/detector.py")
evaluation = load_module("evaluation_under_test", "evaluation/ragas_eval.py")


class FakeCollection:
    def __init__(self, metadata=None):
        self.metadata = metadata if metadata is not None else {"embed_model": SETTINGS.EMBED_MODEL}
        self.rows = {}
        self.delete = Mock(side_effect=AssertionError("Deletion is forbidden"))

    def count(self):
        return len(self.rows)

    def upsert(self, ids, documents, embeddings, metadatas):
        if len(set(ids)) != len(ids):
            raise AssertionError("Duplicate IDs in an upsert")
        for cid, text, metadata in zip(ids, documents, metadatas):
            self.rows[cid] = (text, dict(metadata))

    def get(self, **kwargs):
        return {
            "ids": list(self.rows),
            "documents": [row[0] for row in self.rows.values()],
            "metadatas": [row[1] for row in self.rows.values()],
        }


class CollectionSafetyTests(unittest.TestCase):
    def test_mismatched_or_missing_model_never_deletes_or_creates(self):
        for metadata in ({"embed_model": "old-model"}, {}, None):
            with self.subTest(metadata=metadata):
                coll = Mock(metadata=metadata)
                client = Mock()
                client.get_collection.return_value = coll
                with patch.object(vectorstore, "_collection", None), patch.object(vectorstore, "_client", client):
                    for _ in range(2):
                        with self.assertRaisesRegex(RuntimeError, "No data was changed.*EMBED_MODEL"):
                            vectorstore.get_collection()
                    self.assertIsNone(vectorstore._collection)
                client.create_collection.assert_not_called()
                client.delete_collection.assert_not_called()

    def test_only_explicit_not_found_errors_create(self):
        for error in (NotFoundError, InvalidCollectionException):
            client = Mock()
            client.get_collection.side_effect = error("missing")
            coll = FakeCollection()
            client.create_collection.return_value = coll
            with patch.object(vectorstore, "_collection", None), patch.object(vectorstore, "_client", client):
                self.assertIs(vectorstore.get_collection(), coll)
                self.assertIs(vectorstore.get_collection(), coll)
            client.create_collection.assert_called_once_with(
                vectorstore.COLLECTION,
                metadata={"hnsw:space": "cosine", "embed_model": SETTINGS.EMBED_MODEL},
            )
            client.delete_collection.assert_not_called()

    def test_operational_errors_propagate_unchanged(self):
        for error in (PermissionError("denied"), OSError("corrupt"), ValueError("not found")):
            client = Mock()
            client.get_collection.side_effect = error
            with patch.object(vectorstore, "_collection", None), patch.object(vectorstore, "_client", client):
                with self.assertRaises(type(error)) as caught:
                    vectorstore.get_collection()
                self.assertIs(caught.exception, error)
            client.create_collection.assert_not_called()
            client.delete_collection.assert_not_called()

    def test_collection_initialization_is_thread_safe(self):
        coll = FakeCollection()
        client = Mock()
        client.get_collection.side_effect = NotFoundError("missing")

        def create(*args, **kwargs):
            time.sleep(0.005)
            return coll

        client.create_collection.side_effect = create
        with patch.object(vectorstore, "_collection", None), patch.object(vectorstore, "_client", client):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: vectorstore.get_collection(), range(24)))
        self.assertTrue(all(result is coll for result in results))
        client.create_collection.assert_called_once()

    def test_embedder_initialization_is_thread_safe(self):
        model = object()

        def construct(*args):
            time.sleep(0.005)
            return model

        constructor = Mock(side_effect=construct)
        with patch.object(vectorstore, "_embedder", None), patch.dict(sys.modules, {
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=constructor),
        }):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: vectorstore.get_embedder(), range(24)))
        self.assertTrue(all(result is model for result in results))
        constructor.assert_called_once()


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection()
        self.collection_patch = patch.object(vectorstore, "get_collection", return_value=self.collection)
        self.embed_patch = patch.object(vectorstore, "_embed", side_effect=lambda texts: [[1.0, 0.0] for _ in texts])
        self.collection_patch.start()
        self.embed_patch.start()
        self.addCleanup(self.collection_patch.stop)
        self.addCleanup(self.embed_patch.stop)

    def test_chunk_ids_ignore_dates_and_metadata_order(self):
        vectorstore.add_documents(["Texte"], [{"document_id": "a", "added": "yesterday", "source": "s"}])
        original = set(self.collection.rows)
        vectorstore.add_documents(["Texte"], [{"source": "s", "added": "today", "document_id": "a"}])
        self.assertEqual(set(self.collection.rows), original)
        self.assertEqual(self.collection.count(), 1)

    def test_repeated_identical_chunks_in_one_batch_are_deduplicated(self):
        count = vectorstore.add_documents(["Texte", "Texte"], [{"source": "s"}, {"source": "s"}])
        self.assertEqual(count, 1)
        self.assertEqual(self.collection.count(), 1)

    def test_mismatched_metadata_lengths_fail_before_embedding(self):
        with patch.object(vectorstore, "_embed") as embed:
            with self.assertRaises(ValueError):
                vectorstore.add_documents(["Texte"], [])
        embed.assert_not_called()
        self.assertEqual(self.collection.count(), 0)

    def test_full_document_ingestion_is_idempotent(self):
        text = "Une demande officielle requiert une pièce d'identité. " * 35
        first = ingestion.ingest_text(text, "Titre", "source.txt")
        original = set(self.collection.rows)
        second = ingestion.ingest_text(text, "Titre", "source.txt")
        self.assertEqual(first, second)
        self.assertEqual(set(self.collection.rows), original)
        self.assertEqual(self.collection.count(), first)

    def test_change_after_first_300_characters_creates_non_destructive_version(self):
        prefix = "Procédure administrative détaillée. " * 30
        ingestion.ingest_text(prefix + "Dernier montant 5000.", "Titre", "source.txt")
        original = dict(self.collection.rows)
        ingestion.ingest_text(prefix + "Dernier montant 7000.", "Titre", "source.txt")
        self.assertTrue(set(original).issubset(self.collection.rows))
        self.assertEqual({key: self.collection.rows[key] for key in original}, original)
        self.assertEqual(len({meta["document_id"] for _, meta in self.collection.rows.values()}), 2)
        self.collection.delete.assert_not_called()

    def test_same_title_and_text_but_different_sources_remain_distinct(self):
        ingestion.ingest_text("Petit texte utile", "Titre", "a.txt")
        ingestion.ingest_text("Petit texte utile", "Titre", "b.txt")
        self.assertEqual(self.collection.count(), 2)

    def test_failed_embedding_keeps_existing_version(self):
        ingestion.ingest_text("Version originale", "Titre", "s")
        original = dict(self.collection.rows)
        with patch.object(vectorstore, "_embed", side_effect=RuntimeError("embedding failed")):
            with self.assertRaises(RuntimeError):
                ingestion.ingest_text("Nouvelle version", "Titre", "s")
        self.assertEqual(self.collection.rows, original)
        self.collection.delete.assert_not_called()

    def test_upserts_are_serialized_between_threads(self):
        active = 0
        maximum = 0
        guard = threading.Lock()
        original_upsert = self.collection.upsert

        def upsert(**kwargs):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.002)
                original_upsert(**kwargs)
            finally:
                with guard:
                    active -= 1

        with patch.object(self.collection, "upsert", side_effect=upsert):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: ingestion.ingest_text("Même contenu", "T", "s"), range(16)))
        self.assertEqual(maximum, 1)
        self.assertEqual(self.collection.count(), 1)

    def test_huggingface_signature_maps_columns_and_is_idempotent(self):
        rows = [{"body": "Pièce nécessaire", "name": "Démarche", "origin": "fichier"}]
        load = Mock(return_value=rows)
        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=load)}):
            for _ in range(2):
                self.assertEqual(ingestion.ingest_huggingface_dataset(
                    "owner/data", split="validation", text_column="body",
                    title_column="name", source_column="origin",
                ), 1)
        load.assert_called_with("owner/data", split="validation")
        self.assertEqual(self.collection.count(), 1)
        metadata = next(iter(self.collection.rows.values()))[1]
        self.assertEqual(metadata["title"], "Démarche")
        self.assertEqual(metadata["source"], "fichier")

    def test_huggingface_fallback_identity_does_not_depend_on_row_order(self):
        rows = [{"text": "Premier texte"}, {"text": "Deuxième texte"}]
        load = Mock(side_effect=[rows, list(reversed(rows))])
        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=load)}):
            ingestion.ingest_huggingface_dataset("owner/data", title_column=None, source_column=None)
            original = set(self.collection.rows)
            ingestion.ingest_huggingface_dataset("owner/data", title_column=None, source_column=None)
        self.assertEqual(set(self.collection.rows), original)

    def test_huggingface_missing_dependency_is_actionable(self):
        with patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaisesRegex(RuntimeError, "optional 'datasets'"):
                ingestion.ingest_huggingface_dataset("owner/data")

    def test_huggingface_missing_columns_and_invalid_text_fail(self):
        for rows in ([{"other": "text"}], [{"text": ["not a string"]}]):
            with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=Mock(return_value=rows))}):
                with self.assertRaises(ValueError):
                    ingestion.ingest_huggingface_dataset("owner/data", title_column=None, source_column=None)
        self.assertEqual(self.collection.count(), 0)


class ChunkingTests(unittest.TestCase):
    def test_long_sentence_and_long_word_obey_hard_limit(self):
        for text in ("mot " * 2000, "x" * 5000, "Phrase. " * 500):
            with self.subTest(text=text[:20]):
                chunks = ingestion.chunk_text(text, chunk_size=128, overlap=32)
                self.assertTrue(chunks)
                self.assertTrue(all(0 < len(chunk) <= 128 for chunk in chunks))

    def test_zero_overlap_does_not_copy_entire_previous_chunk(self):
        text = "abcdefghij" * 15
        chunks = ingestion.chunk_text(text, chunk_size=31, overlap=0)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 31 for chunk in chunks))

    def test_tail_is_not_dropped(self):
        text = "x" * 128 + "TAIL"
        self.assertEqual(ingestion.chunk_text(text, 128, 0), ["x" * 128, "TAIL"])

    def test_invalid_size_and_overlap_rejected_even_for_empty_text(self):
        for size, overlap in ((0, 0), (-1, 0), (10, -1), (10, 10), (10, 11), (10, 1.5), (True, 0), (10, False)):
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    ingestion.chunk_text("", size, overlap)

    def test_maximal_overlap_terminates_and_preserves_content(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = ingestion.chunk_text(text, 8, 7)
        self.assertEqual(chunks[0], text[:8])
        self.assertEqual(chunks[-1], text[-8:])
        self.assertLessEqual(len(chunks), len(text))
        self.assertTrue(all(len(c) == 8 for c in chunks))

    def test_sentence_and_markdown_boundaries_are_preserved(self):
        text = "Une phrase complète. Une autre phrase utile.\n\n## Titre\nUne fin."
        chunks = ingestion.chunk_text(text, 46, 0)
        self.assertEqual(chunks[0], "Une phrase complète. Une autre phrase utile.")
        self.assertIn("## Titre", " ".join(chunks))
        self.assertNotIn("## Titre.", ingestion._split_into_sentences(text))
        self.assertEqual("".join("".join(chunks).split()), "".join(text.split()))

    def test_small_bounds_never_discard_nonempty_text(self):
        for size in range(1, 15):
            for overlap in range(size):
                chunks = ingestion.chunk_text("a b c d e f", size, overlap)
                self.assertTrue(all(0 < len(c) <= size for c in chunks))
                self.assertTrue(set("abcdef").issubset(set("".join(chunks))))


class RetrievalTests(unittest.TestCase):
    def test_rrf_is_independent_of_raw_score_scales(self):
        first = hybrid.reciprocal_rank_fusion([(0, 10000), (1, 1)], [(1, 0.9), (0, -0.8)])
        second = hybrid.reciprocal_rank_fusion([(0, 0.01), (1, 0.001)], [(1, 900), (0, -800)])
        self.assertEqual(first, second)
        self.assertAlmostEqual(dict(first)[0], 0.5 / 61 + 0.5 / 62)

    def test_rrf_missing_ranks_and_zero_weights(self):
        self.assertEqual(hybrid.reciprocal_rank_fusion([(0, 5)], [(1, 8)], alpha=1), [(0, 1 / 61)])
        self.assertEqual(hybrid.reciprocal_rank_fusion([(0, 5)], [(1, 8)], alpha=0), [(1, 1 / 61)])
        with self.assertRaises(ValueError):
            hybrid.reciprocal_rank_fusion([], [], alpha=1.1)

    def test_hybrid_class_uses_rrf(self):
        retriever = hybrid.HybridRetriever(["a", "b"])
        with patch.object(retriever, "_bm25_search", return_value=[(0, 10000), (1, 1)]), patch.object(retriever, "_vector_search", return_value=[(1, 0.9)]):
            result = retriever.search("q", k=2)
        self.assertEqual(result[0][0], 1)
        self.assertLess(result[0][1], 0.02)

    def test_empty_and_punctuation_only_corpora_are_safe(self):
        for documents in ([], ["", "...", " "]):
            retriever = hybrid.HybridRetriever(documents)
            self.assertEqual(retriever.search("question"), [])
        self.assertEqual(hybrid.tokenize("Où, déposer l'identité ?"), ["ou", "deposer", "l", "identite"])

    def test_vectorstore_hybrid_preserves_ids_and_metadata_with_rrf(self):
        coll = FakeCollection()
        coll.rows = {"a": ("alpha", {"source": "a"}), "b": ("beta", {"source": "b"})}
        with patch.object(vectorstore, "get_collection", return_value=coll), patch.object(vectorstore, "vector_search", return_value=[("b", "beta", 0.9, {"source": "b"})]), patch.object(hybrid.HybridRetriever, "_bm25_search", return_value=[(0, 10000), (1, 1)]), patch.dict(sys.modules, {"retrieval.hybrid": hybrid}):
            result = vectorstore.hybrid_search("q", k=2)
        self.assertEqual(result[0][0], "b")
        self.assertEqual(result[0][1], "beta")
        self.assertEqual(result[0][3], {"source": "b"})
        self.assertAlmostEqual(result[0][2], 0.4 / 62 + 0.6 / 61)

    def test_vector_search_caps_requested_count(self):
        coll = Mock()
        coll.count.return_value = 1
        coll.query.return_value = {"documents": [["d"]], "metadatas": [[{"source": "s"}]], "distances": [[0.2]], "ids": [["id"]]}
        with patch.object(vectorstore, "get_collection", return_value=coll), patch.object(vectorstore, "_embed_single", return_value=[1.0]):
            result = vectorstore.vector_search("q", k=20)
        self.assertEqual(coll.query.call_args.kwargs["n_results"], 1)
        self.assertAlmostEqual(result[0][2], 0.8)

    def test_mmr_reuses_candidates_and_deduplicates_text_and_ids(self):
        candidates = [
            ("a", "first", 9, {"source": "s"}),
            ("b", " first ", 8, {}),
            ("a", "different", 7, {}),
            ("c", "second", 6, {"source": "t"}),
        ]
        model = Mock()
        model.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] if text in {"q", "first"} else [0.0, 1.0] for text in texts]
        with patch.object(vectorstore, "hybrid_search") as search, patch.object(vectorstore, "get_embedder", return_value=model):
            result = vectorstore.mmr_search("q", k=8, candidates=candidates)
        search.assert_not_called()
        self.assertEqual([row[0] for row in result], ["a", "c"])
        self.assertEqual(result[0][3], {"source": "s"})
        self.assertEqual(len(candidates), 4)

    def test_empty_precomputed_candidates_never_trigger_search(self):
        with patch.object(vectorstore, "hybrid_search") as search, patch.object(vectorstore, "get_embedder") as embedder:
            self.assertEqual(vectorstore.mmr_search("q", candidates=[]), [])
        search.assert_not_called()
        embedder.assert_not_called()

    def test_mmr_rejects_invalid_lambda(self):
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                vectorstore.mmr_search("q", lambda_mmr=value, candidates=[])

    def test_reranker_retains_negative_raw_scores(self):
        model = Mock()
        model.predict.return_value = [-9.0, -0.8, 3.2]
        with patch.object(reranker, "load_model", return_value=model):
            result = reranker.rerank("q", ["a", "b", "c"], top_k=3)
        self.assertEqual(result, [(2, 3.2, "c"), (1, -0.8, "b"), (0, -9.0, "a")])

    def test_reranker_zero_top_k_does_not_load_model(self):
        with patch.object(reranker, "load_model") as load:
            self.assertEqual(reranker.rerank("q", ["a"], top_k=0), [])
        load.assert_not_called()


class LanguageAndEvaluationTests(unittest.TestCase):
    def test_known_wolof_thanks_and_french_false_positive(self):
        for text in ("jerejef", "Jërëjëf !", "jërejëf", "je\u0308reje\u0308f"):
            self.assertEqual(detector.detect_language(text), "wo")
        self.assertEqual(detector.detect_language("Comment obtenir une carte à la mairie ?"), "fr")

    def test_accentless_location_intent_without_bare_ou_false_positive(self):
        self.assertEqual(router.detect_intent("Bonjour, ou est la pharmacie ?"), "orientation")
        self.assertEqual(router.detect_intent("OU   SONT les locaux ?"), "orientation")
        self.assertEqual(router.detect_intent("Faut-il un original ou une copie ?"), "procedure")

    def test_actual_abstention_phrases_are_detected(self):
        for text in (
            "Je n'ai pas trouvé cette information dans ma base documentaire.",
            "Je n’ai pas trouvé d’information dans la base documentaire.",
            "INFORMATION_ABSENTE", "Xamuma tontu.",
        ):
            self.assertTrue(evaluation._is_abstention(text), text)
        self.assertFalse(evaluation._is_abstention("La mairie délivre ce document."))
        self.assertFalse(evaluation._is_abstention(""))

    def test_evaluation_forces_case_language_and_uses_full_chunks(self):
        full = "Début du contexte. " * 30 + "Le coût exact est 7000 francs."
        answer = Mock(return_value={
            "response": "7000 francs", "trace": {
                "retrieval": {"chunks": [{"text": full}], "n_reranked": 1},
            },
        })
        with patch.dict(sys.modules, {"pipeline": types.SimpleNamespace(answer=answer)}):
            result = evaluation.evaluate_single(evaluation.TestCase("Prix ?", "7000 francs", language="wo"))
        self.assertEqual(answer.call_args.kwargs["input_lang"], "wo")
        self.assertEqual(result.retrieved_chunks, [full])
        self.assertEqual(result.context_recall, 1.0)
        self.assertEqual(asdict(result)["evaluation_method"], "local_lexical_heuristics_not_ragas")

    def test_evaluation_falls_back_to_full_context_instead_of_preview(self):
        full = "Un très long texte " * 30 + "référence nécessaire"
        answer = Mock(return_value={"response": "référence nécessaire", "trace": {
            "context": full, "retrieval": {"chunks": [{"text": full[:200] + "…"}]},
        }})
        with patch.dict(sys.modules, {"pipeline": types.SimpleNamespace(answer=answer)}):
            result = evaluation.evaluate_single(evaluation.TestCase("Question", "référence nécessaire"))
        self.assertEqual(result.retrieved_chunks, [full])
        self.assertEqual(result.context_source, "full_context_without_chunk_boundaries")
        self.assertEqual(result.context_recall, 1.0)

    def test_wolof_abstention_can_use_french_source_response(self):
        answer = Mock(return_value={
            "response": "Traduction wolof", "response_fr": "Je n'ai pas trouvé cette information dans ma base documentaire.",
            "trace": {},
        })
        with patch.dict(sys.modules, {"pipeline": types.SimpleNamespace(answer=answer)}):
            result = evaluation.evaluate_single(evaluation.TestCase("q", "", language="wo", category="absente"))
        self.assertTrue(result.abstention_correct)
        self.assertIsNone(result.faithfulness)


if __name__ == "__main__":
    unittest.main()
