"""Mémoire de conversation en RAM pour TontumaBot V3.

Stocke l'historique des échanges par session_id, avec :
  - une limite de `max_messages` messages conservés (troncature aux derniers) ;
  - une durée de vie `duree_vie_s` au-delà de laquelle la session est nettoyée.

Aucune persistance disque : la mémoire est volatile et par process.
"""

import time
from threading import RLock

from config import settings


class SessionLimitReached(ValueError):
    """La session a atteint son nombre maximal de questions."""


class SessionExpired(ValueError):
    """La session a expiré par inactivité et doit être redémarrée."""


class MemoireConversation:
    def __init__(
        self,
        max_questions: int = 5,
        max_messages_context: int = 10,
        duree_vie_s: int = 120,
    ):
        self.max_questions = max_questions
        self.max_messages_context = max_messages_context
        self.duree_vie_s = duree_vie_s
        self._sessions: dict[str, dict] = {}
        self._sessions_fermees: set[str] = set()
        self._sessions_expirees: set[str] = set()
        self._lock = RLock()

    def _session_expiree(self, session: dict) -> bool:
        return (
            time.time() - session["derniere_activite"]
            > self.duree_vie_s
        )

    def _nettoyer_sessions_expirees(self) -> None:
        now = time.time()

        sessions_expirees = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session["derniere_activite"]
            > self.duree_vie_s
        ]

        for session_id in sessions_expirees:
            del self._sessions[session_id]
            self._sessions_expirees.add(session_id)

    def demarrer_session(self, session_id: str) -> dict:
        with self._lock:
            self._nettoyer_sessions_expirees()
            if session_id in self._sessions_fermees:
                raise SessionLimitReached("Limite de questions atteinte pour cette session.")

            session = {
                "messages": [],
                "questions_utilisees": 0,
                "max_questions": self.max_questions,
                "derniere_activite": time.time(),
                "statut": "active",
            }

            self._sessions[session_id] = session

            return {
                "session_id": session_id,
                "statut": "active",
                "questions_restantes": self.max_questions,
            }

    def historique_pour_question(self, session_id: str) -> list[dict]:
        """Crée la première session et refuse une session déjà clôturée ou expirée."""
        with self._lock:
            self._nettoyer_sessions_expirees()
            if session_id in self._sessions_fermees:
                raise SessionLimitReached("Limite de questions atteinte pour cette session.")
            if session_id in self._sessions_expirees:
                raise SessionExpired(
                    f"Votre conversation a expiré après {self.duree_vie_s} secondes d'inactivité. "
                    "Une nouvelle session doit être démarrée."
                )
            if session_id not in self._sessions:
                self.demarrer_session(session_id)
            return list(self._sessions[session_id]["messages"])

    def peut_poser_question(self, session_id: str) -> bool:
        with self._lock:
            self._nettoyer_sessions_expirees()

            session = self._sessions.get(session_id)

            if session is None:
                return False

            return session["questions_utilisees"] < self.max_questions

    def ajouter_tour(
        self,
        session_id: str,
        question: str,
        reponse: str,
    ) -> dict:
        with self._lock:
            self._nettoyer_sessions_expirees()

            session = self._sessions.get(session_id)

            if session is None:
                if session_id in self._sessions_fermees:
                    raise SessionLimitReached("Limite de questions atteinte pour cette session.")
                raise SessionExpired(
                    f"Votre conversation a expiré après {self.duree_vie_s} secondes d'inactivité. "
                    "Une nouvelle session doit être démarrée."
                )

            if session["questions_utilisees"] >= self.max_questions:
                self._sessions_fermees.add(session_id)
                self._sessions.pop(session_id, None)

                raise SessionLimitReached("Limite de questions atteinte.")

            session["messages"].append({"role": "user", "content": question})

            session["messages"].append({"role": "assistant", "content": reponse})

            session["messages"] = session["messages"][-self.max_messages_context:]

            session["questions_utilisees"] += 1
            session["derniere_activite"] = time.time()

            questions_restantes = self.max_questions - session["questions_utilisees"]

            if questions_restantes == 0:
                self._sessions_fermees.add(session_id)
                self._sessions.pop(session_id, None)

                return {
                    "statut": "limite_atteinte", "questions_restantes": 0, "session_fermee": True,
                    "message": (
                        f"Vous avez atteint la limite de {self.max_questions} questions. "
                        "La conversation a été réinitialisée. Démarrez une nouvelle session."
                    ),
                }

            return {"statut": "active", "questions_restantes": questions_restantes, "session_fermee": False}

    def obtenir_historique(
        self,
        session_id: str,
    ) -> list[dict]:
        self._nettoyer_sessions_expirees()

        session = self._sessions.get(session_id)

        if not session:
            return []

        return list(session["messages"])

    def statut_session(self, session_id: str) -> dict:
        self._nettoyer_sessions_expirees()

        if session_id in self._sessions_fermees:
            return {
                "statut": "limite_atteinte",
                "questions_utilisees": self.max_questions,
                "questions_restantes": 0,
                "secondes_restantes": 0,
            }

        if session_id in self._sessions_expirees:
            return {
                "statut": "expiree",
                "questions_restantes": 0,
                "secondes_restantes": 0,
            }

        session = self._sessions.get(session_id)

        if session is None:
            return {
                "statut": "inexistante",
                "questions_restantes": 0,
            }

        secondes_restantes = max(
            0,
            int(
                self.duree_vie_s
                - (time.time() - session["derniere_activite"])
            ),
        )

        return {
            "statut": session["statut"],
            "questions_utilisees": session[
                "questions_utilisees"
            ],
            "questions_restantes": (
                self.max_questions
                - session["questions_utilisees"]
            ),
            "secondes_restantes": secondes_restantes,
        }

    def effacer_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._sessions_fermees.discard(session_id)
            self._sessions_expirees.discard(session_id)


memoire = MemoireConversation(
    max_questions=settings.MAX_QUESTIONS_PER_SESSION,
    max_messages_context=settings.MEMORY_MAX_TURNS * 2,
    duree_vie_s=settings.SESSION_INACTIVITY_SECONDS,
)