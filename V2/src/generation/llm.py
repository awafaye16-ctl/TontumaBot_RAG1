"""Générateur de réponse LLM : Groq ou Gemini."""
import re
from config import settings

SYSTEM_PROMPT = (
    "Tu es un assistant administratif sénégalais. Réponds en français, "
    "uniquement à partir du contexte fourni. Sois court et précis. "
    "Si l'information n'y est pas, dis que tu ne sais pas. "
    "Ne jamais inclure de raisonnement ou de balises <think> dans ta réponse."
)


def _groq_generate(question_fr: str, context_fr: str) -> str:
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexte :\n{context_fr}\n\nQuestion : {question_fr}"},
        ],
        temperature=0.2,
    )
    raw = completion.choices[0].message.content
    return _strip_think(raw)


def _gemini_generate(question_fr: str, context_fr: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"{SYSTEM_PROMPT}\n\nContexte :\n{context_fr}\n\nQuestion : {question_fr}"
    return model.generate_content(prompt).text


def _strip_think(text: str) -> str:
    """Supprime les balises <think>...</think> des réponses LLM (qwen, etc.)."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _fallback(question_fr: str, context_fr: str) -> str:
    if not context_fr.strip():
        return "Je n'ai pas trouvé d'information dans la base documentaire."
    return context_fr.split("\n\n")[0]


def generate(question_fr: str, context_fr: str, provider: str = "groq") -> str:
    if provider == "gemini":
        if settings.GEMINI_API_KEY:
            return _gemini_generate(question_fr, context_fr)
        return _fallback(question_fr, context_fr)
    if settings.GROQ_API_KEY:
        return _groq_generate(question_fr, context_fr)
    return _fallback(question_fr, context_fr)
