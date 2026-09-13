import time

from src.config import RETRIEVAL_PHASES
from src.evaluating.hardware.evaluate_hardware_consume import measure_operation
from src.generation.generate_answer import generate_answer
from src.retrieval.retriever import embed_text, search_qdrant, fetch_chunks_from_mongo, build_context
from src.translator.query_translation import translate_question, translate_answer_back
from src.utils.retrieval_error_handler import build_retrieval_result_by_first_level_error


def run_trag_retrieval(question: str, question_lang: str, doc_lang:str, generate=False):

    t0 = time.perf_counter()
    error = None
    hardware = []
    
    translate_result, translate_metrics = measure_operation(
        lambda: translate_question(question, question_lang, doc_lang)
    )
    translated_question, query_translation_time = translate_result
    translate_metrics.update({"rag_strategy": "tRAG"})
    translate_metrics.update({"phase": RETRIEVAL_PHASES[0]})
    hardware.append(translate_metrics)
    if translated_question["success"] is False:
        return {
            "question": question,
            "translated_question": "",
            "answer": "",
            "retrieval_type": "tRAG",
            "error": translated_question["error"],
            "hits": [],
            "chunks": [],
            "translated_documents": [],
            "context": [],
            "retrieval_time": time.perf_counter() - t0 - query_translation_time,
            "generate_answer_time": query_translation_time,
            "translate_time": 0.0,
            "total_time": time.perf_counter() - t0
        }, hardware
    
    vector = embed_text(translated_question["response"])
    hits = search_qdrant(vector, lang=doc_lang)

    if hits is None or len(hits) == 0:
        retrieval_time = time.perf_counter() - t0
        hits = []
        return build_retrieval_result_by_first_level_error(translated_question["response"], retrieval_time, "tRAG", hits, hardware, t0, occurred_error_type=1, query_translation_time=query_translation_time)
    chunks = fetch_chunks_from_mongo(hits)
    if chunks is None or len(chunks) == 0:
        retrieval_time = time.perf_counter() - t0
        return build_retrieval_result_by_first_level_error(translated_question["response"], retrieval_time, "tRAG", hits, hardware, t0, occurred_error_type=2,query_translation_time=query_translation_time)

    retrieval_time = time.perf_counter() - t0 - query_translation_time

    result = {
        "question": question,
        "translated_question": translated_question["response"],
        "answer": "",
        "retrieval_type": "tRAG",
        "error": error,
        "hits": hits,
        "chunks": chunks,
        "translated_documents": [],
        "context": [],
        "retrieval_time": retrieval_time,
        "generate_answer_time": 0.0,
        "translate_time": query_translation_time,
        "total_time": time.perf_counter() - t0
    }

    if generate:
        context, context_array = build_context(chunks)
        generate_result, generate_metrics = measure_operation(
            lambda: generate_answer(translated_question["response"], context, doc_lang)
        )
        response, generate_answer_time = generate_result
        generate_metrics.update({"rag_strategy": "tRAG"})
        generate_metrics.update({"phase": RETRIEVAL_PHASES[3]})
        hardware.append(generate_metrics)
        if response["success"] is False:
            result["answer"] = str(response["response"])
            result["error"] = response["error"]
        else:
            translate_back_result, translate_back_metrics = measure_operation(
                lambda: translate_answer_back(response["response"], doc_lang, question_lang)
            )
            translated_answer, answer_translation_time = translate_back_result
            translate_back_metrics.update({"rag_strategy": "tRAG"})
            translate_back_metrics.update({"phase": RETRIEVAL_PHASES[2]})
            hardware.append(translate_back_metrics)
            result["answer"] = str(translated_answer["response"])
            result["error"] = translated_answer["error"]
            result["translate_time"] = query_translation_time + answer_translation_time
        result['context'] = context_array
        result['generate_answer_time'] = generate_answer_time
        result['total_time'] = time.perf_counter() - t0

    return result, hardware
