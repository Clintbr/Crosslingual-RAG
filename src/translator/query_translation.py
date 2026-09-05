import time
import requests

from src.config import OLLAMA_URL, TRANSLATION_MODEL


def translate_question(question: str, original_lang: str, target_lang: str):
    start = time.perf_counter()

    prompt = f"""
        Translate the following question from {original_lang} to {target_lang}.
        Return ONLY the translated question.
        
        Question:
        {question}
    """

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": TRANSLATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # set to null to avoid high hallucination
                "num_ctx": 4096  # big window for more context
            }
        },
        timeout=120
    )

    response.raise_for_status()

    translation = response.json().get("response", "").strip()

    return translation, time.perf_counter() - start

def translate_answer_back():
    return