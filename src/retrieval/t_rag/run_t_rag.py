import time

from src.generation.generate_answer import generate_answer
from src.retrieval.retriever import embed_text, search_qdrant, fetch_chunks_from_mongo, build_context
from src.translator.query_translation import translate_question, translate_answer_back
from src.utils.retrieval_error_handler import build_retrieval_result_by_first_level_error


def run_trag_retrieval(question: str, question_lang: str, doc_lang:str, generate=False):

    t0 = time.perf_counter()
    error = None
    translated_question, query_translation_time = translate_question(question, question_lang, doc_lang)
    if translated_question["success"] is False:
        return {
            "question": question,
            "answer": "",
            "retrieval_type": "multiRAG",
            "error": translated_question["error"],
            "hits": [],
            "chunks": [],
            "context": "",
            "retrieval_time": time.perf_counter() - t0 - query_translation_time,
            "generate_answer_time": query_translation_time,
            "translate_time": 0.0
        }
    
    vector = embed_text(translated_question["response"])
    hits = search_qdrant(vector, lang=doc_lang)

    if hits.empty:
        retrieval_time = time.perf_counter() - t0
        hits = []
        return build_retrieval_result_by_first_level_error(translated_question["response"], retrieval_time, "tRAG", hits, occurred_error_type=1)
    chunks = fetch_chunks_from_mongo(hits)
    if chunks.empty:
        retrieval_time = time.perf_counter() - t0
        return build_retrieval_result_by_first_level_error(translated_question["response"], retrieval_time, "tRAG", hits, occurred_error_type=2)

    retrieval_time = time.perf_counter() - t0 - query_translation_time

    result = {
        "question": translated_question["response"],
        "answer": "",
        "retrieval_type": "tRAG",
        "error": error,
        "hits": hits,
        "chunks": chunks,
        "context": "",
        "retrieval_time": retrieval_time,
        "generate_answer_time": 0.0,
        "translate_time": query_translation_time
    }

    if generate:
        context = build_context(chunks)
        response, generate_answer_time = generate_answer(translated_question["response"], context, doc_lang)
        if response["success"] is False:
            result["answer"] = response["response"]
            result["error"] = response["error"]
        else:
            start = time.perf_counter()
            translated_answer, answer_translation_time = translate_answer_back(response["response"], doc_lang, question_lang)
            result["answer"] = translated_answer["response"]
            result["error"] = translated_answer["error"]
            result["translate_time"] = (query_translation_time | 0.0) + (answer_translation_time | time.perf_counter() - start)
        result['context'] = context
        result['generate_answer_time'] = generate_answer_time | 0.0

    return result
