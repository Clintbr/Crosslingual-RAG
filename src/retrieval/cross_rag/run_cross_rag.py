import time

from src.generation.generate_answer import generate_answer
from src.retrieval.retriever import embed_text, search_qdrant, fetch_chunks_from_mongo, build_context
from src.translator.document_translation import translate_retrieved_doc
from src.utils.retrieval_error_handler import build_retrieval_result_by_first_level_error


def run_cross_retrieval(question: str, question_lang: str, generate=False):

    t0 = time.perf_counter()
    error = None
    vector = embed_text(question)
    hits = search_qdrant(vector)

    if hits.empty:
        retrieval_time = time.perf_counter() - t0
        hits = []
        return build_retrieval_result_by_first_level_error(question, retrieval_time, 'crossRAG', hits, occurred_error_type=1)
    chunks = fetch_chunks_from_mongo(hits)
    if chunks.empty:
        retrieval_time = time.perf_counter() - t0
        return build_retrieval_result_by_first_level_error(question, retrieval_time, 'crossRAG', hits, occurred_error_type=2)

    retrieval_time = time.perf_counter() - t0

    result = {
        "question": question,
        "answer": "",
        "retrieval_type": "crossRAG",
        "error": error,
        "hits": hits,
        "chunks": chunks,
        "context": "",
        "retrieval_time": retrieval_time,
        "generate_answer_time": 0.0,
        "translate_time": 0.0
    }

    if generate:
        start = time.perf_counter()
        translated_chunks, translate_time, has_error = translate_retrieved_doc(chunks, question_lang)
        if has_error:
            result['generate_answer_time'] = time.perf_counter() - start
            result["error"] = translated_chunks["error"]
        else:
            context = build_context(translated_chunks, translated=True)
            response, generate_answer_time = generate_answer(question, context, question_lang)
            result['context'] = context
            result['generate_answer_time'] = generate_answer_time | time.perf_counter() - start
            result["answer"] = response["response"]
            result["error"] = response["error"]
        result["translate_time"] = translate_time

    return result