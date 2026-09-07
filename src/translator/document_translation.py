import time

from src.api.ollama_api import send_to_ollama
from src.config import TRANSLATION_MODEL, RETRIEVAL_PHASES
from src.prompts.prompt_templates import prompt_translate_text

# Document - Translation
def translate_retrieved_doc(documents: [], target_lang: str):
    translated_documents = documents
    start = time.perf_counter()
    has_error = False

    for document in translated_documents:
        start = time.perf_counter()
        prompt = prompt_translate_text(document.get("lang"), target_lang, document.get("content"))
        response = send_to_ollama(TRANSLATION_MODEL, prompt, RETRIEVAL_PHASES[1])

        if response.get("success") is False :
            has_error = True
            return response, time.perf_counter() - start, has_error
        else:
            translated_content = response.get("response")
            document.update({"translation": translated_content})

    return translated_documents, time.perf_counter() - start, has_error

