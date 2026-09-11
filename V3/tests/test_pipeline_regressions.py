import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pipeline


class PipelineTests(unittest.TestCase):
    def test_business_question_is_not_greeting(self):
        self.assertFalse(pipeline._is_greeting("Bonjour, où est la pharmacie ?"))
        self.assertFalse(pipeline._is_greeting("Historique de la mairie"))
        self.assertTrue(pipeline._is_greeting("Bonjour !"))

    def test_forced_wolof_greeting_is_respected(self):
        with patch.object(pipeline, "french_to_wolof", return_value=("Salaam", 0.1)), patch.object(pipeline.vectorstore, "count") as count:
            result = pipeline.answer("Bonjour", input_lang="wo")
        self.assertEqual(result["trace"]["input_lang"], "wo")
        self.assertEqual(result["response_wo"], "Salaam")
        count.assert_not_called()

    def test_invalid_language_is_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.answer("Bonjour", input_lang="xx")

    def test_rag_reuses_candidates_and_keeps_sources(self):
        candidates = [("c1", "La pharmacie est dans le bâtiment A.", 0.4, {"source": "orientation.pdf", "document_id": "d1"})]
        with patch.object(pipeline.vectorstore, "hybrid_search", return_value=candidates), patch.object(pipeline.vectorstore, "mmr_search", return_value=candidates) as mmr, patch.object(pipeline, "rerank", return_value=[(0, 3.0, candidates[0][1])]):
            context, trace = pipeline._rag_pipeline("Où est la pharmacie ?", "orientation")
        self.assertIs(mmr.call_args.kwargs["candidates"], candidates)
        self.assertEqual(trace["chunks"][0]["source"], "orientation.pdf")
        self.assertEqual(trace["chunks"][0]["id"], "c1")
        self.assertIn(candidates[0][1], context)

    def test_empty_generation_never_translates_into_fake_success(self):
        with patch.object(pipeline, "wolof_to_french", return_value=("Où est la pharmacie ?", 0.1)), patch.object(pipeline.vectorstore, "count", return_value=1), patch.object(pipeline, "_rag_pipeline", return_value=("Pharmacie bâtiment A", {"chunks": [], "n_reranked": 1})), patch.object(pipeline, "generate", return_value=""), patch.object(pipeline, "french_to_wolof") as translate:
            with self.assertRaises(RuntimeError):
                pipeline.answer("Ana pharmacie bi ?", input_lang="wo")
        translate.assert_not_called()

    def test_stream_and_rest_share_preparation_and_finalization(self):
        async def fake_stream(*args, **kwargs):
            yield "La pharmacie "
            yield "est dans A."

        with patch.object(pipeline.vectorstore, "count", return_value=1), patch.object(pipeline, "_rag_pipeline", return_value=("Pharmacie dans A", {"chunks": [{"id": "c1", "source": "test.pdf"}], "n_reranked": 1})), patch.object(pipeline, "generate", return_value="La pharmacie est dans A."), patch.object(pipeline, "generate_stream", fake_stream):
            rest = pipeline.answer("Ana pharmacie bi ?", input_lang="fr")

            async def collect():
                return [event async for event in pipeline.answer_stream("Ana pharmacie bi ?", input_lang="fr")]

            events = asyncio.run(collect())
        self.assertEqual(events[0], {"type": "lang", "lang": "fr"})
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["response"], rest["response"])
        self.assertEqual(events[-1]["trace"]["context"], rest["trace"]["context"])
        self.assertEqual(events[-1]["sources"], rest["sources"])

    def test_session_state_is_exposed_and_limit_enforced(self):
        from memory.conversation import MemoireConversation, SessionLimitReached
        test_memoire = MemoireConversation(max_questions=2, duree_vie_s=60)
        with patch.object(pipeline, "memoire", test_memoire), \
             patch.object(pipeline.vectorstore, "count", return_value=1), \
             patch.object(pipeline, "_rag_pipeline", return_value=("Pharmacie dans A", {"chunks": [], "n_reranked": 1})), \
             patch.object(pipeline, "generate", return_value="La pharmacie est dans A."):
            first = pipeline.answer("Où est la pharmacie ?", input_lang="fr", session_id="sess-1")
            self.assertEqual(first["session"], {"statut": "active", "questions_restantes": 1, "session_fermee": False})
            second = pipeline.answer("Quels horaires ?", input_lang="fr", session_id="sess-1")
            self.assertTrue(second["session"]["session_fermee"])
            self.assertIn("2 questions", second["session"]["message"])
            with self.assertRaises(SessionLimitReached):
                pipeline.answer("Une autre ?", input_lang="fr", session_id="sess-1")

    def test_prepare_translates_wolof_even_when_history_present(self):
        # Bug réel : si la reformulation échoue et retombe sur le texte brut, la
        # traduction WO->FR ne doit pas être court-circuitée pour autant.
        with patch.object(pipeline, "wolof_to_french", return_value=("Où est la pharmacie ?", 0.1)) as wo_fr, \
             patch.object(pipeline.vectorstore, "count", return_value=1), \
             patch.object(pipeline, "_rag_pipeline", return_value=("contexte", {"chunks": [], "n_reranked": 1})):
            question, context, trace = pipeline._prepare(
                "Ana farmasi bi ?", "groq", False, None, None, None, "wo",
                question_recherche=None, historique=[{"role": "user", "content": "Bonjour"}])
        wo_fr.assert_called_once()
        self.assertEqual(question, "Où est la pharmacie ?")
        self.assertIn("wolof_to_french", trace)

    def test_reformulation_failure_returns_none_not_raw_text(self):
        with patch("generation.llm._groq_generate", side_effect=RuntimeError("boom")):
            self.assertIsNone(pipeline.reformuler_question("Ana farmasi bi ?", [{"role": "user", "content": "Bonjour"}]))
        self.assertIsNone(pipeline.reformuler_question("Une question", []))

    def test_session_history_stores_french_not_wolof(self):
        from memory.conversation import MemoireConversation
        test_memoire = MemoireConversation(max_questions=5, duree_vie_s=120)
        with patch.object(pipeline, "memoire", test_memoire), \
             patch.object(pipeline, "wolof_to_french", return_value=("Où est la pharmacie ?", 0.1)), \
             patch.object(pipeline, "french_to_wolof", return_value=("Fii la mel ...", 0.1)), \
             patch.object(pipeline.vectorstore, "count", return_value=1), \
             patch.object(pipeline, "_rag_pipeline", return_value=("contexte", {"chunks": [], "n_reranked": 1})), \
             patch.object(pipeline, "generate", return_value="La pharmacie est juste ici."):
            pipeline.answer("Ana farmasi bi ?", input_lang="wo", session_id="wo-hist-test")
        historique = test_memoire.obtenir_historique("wo-hist-test")
        self.assertEqual(historique[0]["content"], "Où est la pharmacie ?")
        self.assertEqual(historique[1]["content"], "La pharmacie est juste ici.")


if __name__ == "__main__":
    unittest.main()
