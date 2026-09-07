# Answer Generation
import time

from src.api.ollama_api import send_to_ollama
from src.config import RETRIEVAL_PHASES, GENERATION_MODEL
from src.prompts.prompt_templates import prompt_generate_response


def generate_answer(question: str, context: str, target_lang: str):
    start = time.perf_counter()
    prompt = prompt_generate_response(target_lang, question, context)
    response = send_to_ollama(GENERATION_MODEL, prompt, RETRIEVAL_PHASES[3])

    return response, time.perf_counter() - start
