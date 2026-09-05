import time

from src.retrieval.retriever import (
    embed_text, search_qdrant, fetch_chunks_from_mongo, generate_answer)
from src.translator.document_translation import translate_retrieved_doc


def run_cross_retrieval(question: str, doc_lang: str, generate=False):

    t0 = time.perf_counter()

    vector = embed_text(question)
    hits = search_qdrant(vector)
    raw_chunks = fetch_chunks_from_mongo(hits)
    chunks, translate_time = translate_retrieved_doc(raw_chunks, doc_lang)

    retrieval_time = time.perf_counter() - t0

    result = {
        "question": question,
        "retrieval_type": "crossRAG",
        "hits": hits,
        "chunks": chunks,
        "retrieval_time": retrieval_time,
        "generate_answer_time": 0.0,
        "translate_time": translate_time
    }

    if generate:
        start = time.perf_counter()
        result["answer"] = generate_answer(question, chunks, doc_lang)
        result["generate_answer_time"] = time.perf_counter() - start

    return result

