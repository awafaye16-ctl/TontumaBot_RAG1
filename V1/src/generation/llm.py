"""Générateur de réponse : LLM via Groq ou Gemini.

La réponse est générée en FRANÇAIS à partir du contexte RAG.
"""
from config import settings

GROQ_API_KEY = settings.GROQ_API_KEY
GEMINI_API_KEY = settings.GEMINI_API_KEY

SYSTEM_PROMPT = (
    "Tu es un assistant administratif sénégalais. Réponds en français, "
    "uniquement à partir du contexte fourni. Si l'information n'y est pas, "
    "dis que tu ne sais pas."
)


def _groq_generate(question_fr: str, context_fr: str) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexte :\n{context_fr}\n\nQuestion : {question_fr}"},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def _gemini_generate(question_fr: str, context_fr: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"{SYSTEM_PROMPT}\n\nContexte :\n{context_fr}\n\nQuestion : {question_fr}"
    return model.generate_content(prompt).text


def _fallback(question_fr: str, context_fr: str) -> str:
    """Mode sans clé API : renvoie le contexte le plus pertinent."""
    if not context_fr.strip():
        return "Je n'ai pas trouvé d'information dans la base documentaire."
    return context_fr.split("\n\n")[0]


def generate(question_fr: str, context_fr: str, provider: str = "groq") -> str:
    """Génère la réponse FR. provider: 'groq' ou 'gemini'.
    Sans clé API configurée, retombe en mode fallback (contexte seul).
    """
    if provider == "gemini":
        if settings.GEMINI_API_KEY:
            return _gemini_generate(question_fr, context_fr)
        return _fallback(question_fr, context_fr)
    if settings.GROQ_API_KEY:
        return _groq_generate(question_fr, context_fr)
    return _fallback(question_fr, context_fr)