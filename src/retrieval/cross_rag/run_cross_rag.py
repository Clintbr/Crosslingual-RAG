"""
The use of lamda function measure_operation here is for evaluating
RAM, VRAM and CPU consumption of intern rag steps like translation
"""

import time

from src.config import RETRIEVAL_PHASES
from src.evaluating.hardware.evaluate_hardware_consume import measure_operation
from src.generation.generate_answer import generate_answer
from src.retrieval.retriever import embed_text, search_qdrant, fetch_chunks_from_mongo, build_context
from src.translator.document_translation import translate_retrieved_doc
from src.utils.retrieval_error_handler import build_retrieval_result_by_first_level_error



def run_cross_retrieval(question: str, question_lang: str, generate=False):
    t0 = time.perf_counter()
    error = None
    vector = embed_text(question)
    hits = search_qdrant(vector)
    hardware = []

    if hits is None or len(hits) == 0:
        retrieval_time = time.perf_counter() - t0
        hits = []
        return build_retrieval_result_by_first_level_error(question, retrieval_time, 'crossRAG', hits, hardware, occurred_error_type=1)
    chunks = fetch_chunks_from_mongo(hits)
    if chunks is None or len(chunks) == 0:
        retrieval_time = time.perf_counter() - t0
        return build_retrieval_result_by_first_level_error(question, retrieval_time, 'crossRAG', hits, hardware, occurred_error_type=2)

    retrieval_time = time.perf_counter() - t0

    result = {
        "question": question,
        "translated_question": "",
        "answer": "",
        "retrieval_type": "crossRAG",
        "error": error,
        "hits": hits,
        "chunks": chunks,
        "translated_documents": [],
        "context": [],
        "retrieval_time": retrieval_time,
        "generate_answer_time": 0.0,
        "translate_time": 0.0,
        "total_time": time.perf_counter() - t0
    }
    if generate:
        translate_result, translate_metrics = measure_operation(
            lambda: translate_retrieved_doc(chunks, question_lang)
        )
        translated_chunks, translate_time, has_error = translate_result
        translate_metrics.update({"rag_strategy": "crossRAG"})
        translate_metrics.update({"phase": RETRIEVAL_PHASES[1]})
        hardware.append(translate_metrics)
        if has_error:
            result["error"] = translated_chunks["error"]
        else:
            context, context_array = build_context(translated_chunks, translated=True)
            generate_result, generate_metrics = measure_operation(
                lambda: generate_answer(question, context, question_lang)
            )
            response, generate_answer_time = generate_result
            generate_metrics.update({"rag_strategy": "crossRAG"})
            generate_metrics.update({"phase": RETRIEVAL_PHASES[3]})
            hardware.append(generate_metrics)
            result["translated_documents"] = translated_chunks
            result["context"] = context_array
            result["generate_answer_time"] = generate_answer_time
            result["answer"] = str(response["response"])
            result["error"] = response["error"]
        result["translate_time"] = translate_time
        result['total_time'] = time.perf_counter() - t0
    return result, hardware