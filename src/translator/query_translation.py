import time

from src.api.ollama_api import send_to_ollama
from src.config import TRANSLATION_MODEL, RETRIEVAL_PHASES
from src.prompts.prompt_templates import prompt_translate_text, prompt_translate_question


def translate_question(question: str, original_lang: str, target_lang: str):
    start = time.perf_counter()
    prompt = prompt_translate_question(original_lang, target_lang, question)
    response = send_to_ollama(TRANSLATION_MODEL, prompt, RETRIEVAL_PHASES[0])

    return response, time.perf_counter() - start

def translate_answer_back(answer: str, original_lang: str, target_lang: str):

    start = time.perf_counter()
    prompt = prompt_translate_text(original_lang, target_lang, answer)
    response = send_to_ollama(TRANSLATION_MODEL, prompt, RETRIEVAL_PHASES[2])

    return response, time.perf_counter() - start