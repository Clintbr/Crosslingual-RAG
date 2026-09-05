from src.config import (
    DE, EN, FR)

## the following prompts are written based on Zero-Shot-Prompting

def print_lang(lang:str):
    if lang == 'en':
        return 'english'
    elif lang == 'fr':
        return 'french'
    else:
        return 'german'

def prompt_translate_question(original_lang, target_lang, question) -> str:
    from_lang = print_lang(original_lang)
    to_lang = print_lang(target_lang)

    return (
        f"You are an expert, native-level translator fluent in both {from_lang} and {to_lang},"
        "specializing in linguistics and cultural nuances.\n"
        f"Your task is to translate the following question from {from_lang} into {to_lang}."
        "Strictly follow these execution rules:"
        "1. Maintain the exact original meaning, intent, and tone of the question.\n"
        f"2. Adapt the sentence structure naturally to fit the native phrasing of {to_lang}, avoiding awkward, literal word-for-word translations.\n"
        f"3. Ensure that it remains a natural sounding question in {to_lang}.\n"
        "4. Output ONLY the final translated question. Do not include any introductions, explanations, notes, or quotation marks.\n\n"
        "Question to translate:"
        f"{question}"
    )

def prompt_translate_text (original_lang, target_lang, text) -> str:
    from_lang = print_lang(original_lang)
    to_lang = print_lang(target_lang)

    return (
        f"You are an expert, native-level translator fluent in both {from_lang} and {to_lang},"
        "specializing in linguistics and cultural nuances.\n"
        f"Your task is to translate the following text from {from_lang} into {to_lang}."
        "Strictly follow these execution rules:"
        "1. Maintain the exact original meaning, intent, and tone of the text.\n"
        f"2. Adapt the sentence structure naturally to fit the native phrasing of {to_lang}, avoiding awkward, literal word-for-word translations.\n"
        "3. Output ONLY the final translated text. Do not include any introductions, explanations, notes, or quotation marks.\n\n"
        "Question to translate:"
        f"{text}"
    )

def prompt_generate_response (target_lang, question, context) -> str:
    to_lang = print_lang(target_lang)

    return (
        "You are an advanced, analytical AI Knowledge Assistant"
        "specializing in precise information retrieval and multilingual synthesis."
        "Your objective is to answer the user's question using ONLY the facts provided in the source text below,"
        f"and you must formulate your entire response exclusively in {to_lang}.\n"
        "Strict execution rules:"
        f"1. MANDATORY LANGUAGE: Write your entire response in {to_lang}. Even if the provided context or the question is in another language, the final output must be 100% in {to_lang}.\n"
        f"2. STRICT TRUTH-FULNESS: Rely solely on the clear facts mentioned in the PROVIDED CONTEXT. Do not assume, extrapolate, or use outside knowledge\n."
        "3. MISSING INFORMATION ALGORITHM: If the context does not contain the answer to the question, do not attempt to guess. Reply exactly with: I cannot find the answer in the provided documents.\n"
        "4. ZERO-SHOT DIRECTNESS: Do not include introductory phrases (like Based on the text...), explanations, or conversational filler. Start directly with the answer.\n\n"

        "PROVIDED CONTEXT:\n"
        f"{context}\n\n"

        "USER QUESTION:\n"
        f"{question}\n\n"

        f"FINAL RESPONSE IN {to_lang}:"
    )