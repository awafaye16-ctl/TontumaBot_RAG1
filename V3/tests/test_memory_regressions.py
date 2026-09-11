import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from memory.conversation import MemoireConversation, SessionExpired, SessionLimitReached


class MemoryTests(unittest.TestCase):
    def test_session_is_created_and_closed_at_configured_limit(self):
        memory = MemoireConversation(max_questions=2, duree_vie_s=60)
        self.assertEqual(memory.historique_pour_question("s"), [])
        memory.ajouter_tour("s", "q1", "r1")
        result = memory.ajouter_tour("s", "q2", "r2")
        self.assertTrue(result["session_fermee"])
        self.assertIn(str(memory.max_questions), result["message"])
        self.assertEqual(memory.statut_session("s")["questions_restantes"], 0)
        with self.assertRaises(SessionLimitReached):
            memory.historique_pour_question("s")

    def test_inactivity_expires_session_explicitly(self):
        memory = MemoireConversation(max_questions=2, duree_vie_s=1)
        memory.historique_pour_question("s")
        memory._sessions["s"]["derniere_activite"] = time.time() - 2
        self.assertEqual(memory.statut_session("s")["statut"], "expiree")
        with self.assertRaises(SessionExpired):
            memory.historique_pour_question("s")

    def test_unknown_session_status_is_inexistante(self):
        memory = MemoireConversation(max_questions=2, duree_vie_s=60)
        self.assertEqual(memory.statut_session("never-seen")["statut"], "inexistante")

    def test_expiry_during_request_processing_raises_session_expired_not_value_error(self):
        # Une requête lente (chargement de modèle, LLM) peut dépasser le TTL entre la
        # lecture de l'historique et la sauvegarde du tour : doit rester un signal
        # d'expiration explicite (410), pas une ValueError générique (400).
        memory = MemoireConversation(max_questions=2, duree_vie_s=1)
        memory.historique_pour_question("s")
        memory._sessions["s"]["derniere_activite"] = time.time() - 2
        with self.assertRaises(SessionExpired):
            memory.ajouter_tour("s", "q1", "r1")


if __name__ == "__main__":
    unittest.main()