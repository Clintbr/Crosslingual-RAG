"""
    document_translation.py
    this function is used by crossRAG for the multiple translation of the retrieved context
"""

import time
from copy import deepcopy

from src.api.ollama_api import send_to_ollama
from src.config import TRANSLATION_MODEL, RETRIEVAL_PHASES
from src.prompts.prompt_templates import prompt_translate_text

# Document - Translation
def translate_retrieved_doc(documents: list, target_lang: str):
    translated_documents = deepcopy(documents)
    start_0 = time.perf_counter()
    has_error = False

    for document in translated_documents:
        start_1 = time.perf_counter()
        if document.get("lang") == target_lang:
            document.update({"translation": document.get("content")})
            document.update({"single_translation_time": time.perf_counter() - start_1})
        else:
            prompt = prompt_translate_text(document.get("lang"), target_lang, document.get("content"))
            response = send_to_ollama(TRANSLATION_MODEL, prompt, RETRIEVAL_PHASES[1])

            if response.get("success") is False :
                has_error = True
                return response, time.perf_counter() - start_0, has_error
            else:
                translated_content = response.get("response")
                document.update({"translation": translated_content})
                document.update({"single_translation_time": time.perf_counter() - start_1})

    return translated_documents, time.perf_counter() - start_0, has_error

