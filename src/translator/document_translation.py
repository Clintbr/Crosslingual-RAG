# Document - Translation
import time
import requests

from src.config import OLLAMA_URL, TRANSLATION_MODEL


def translate_retrieved_doc(documents: [], target_lang: str):
    translated_documents = documents
    start = time.perf_counter()

    for document in translated_documents:
        prompt = f"""
            Translate the following Text from {document.get("lang")} to {target_lang}.
            Return ONLY the translated question.
            
            TEXT:
            {document.get("content")}
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
        translated_documents.update({"translation": translation})

    return translated_documents, time.perf_counter() - start

