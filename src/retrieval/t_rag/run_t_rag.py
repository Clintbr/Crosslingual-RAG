import time

from src.retrieval.retriever import (
    embed_text, search_qdrant, fetch_chunks_from_mongo, generate_answer)
from src.translator.query_translation import translate_question


def run_trag_retrieval(question: str, doc_lang: str, generate=False):

    t0 = time.perf_counter()

    translated_question, translate_time = translate_question(question, doc_lang)

    vector = embed_text(translated_question)
    hits = search_qdrant(vector, doc_lang)
    chunks = fetch_chunks_from_mongo(hits)

    retrieval_time = time.perf_counter() - t0

    result = {
        "question": question,
        "translated_question": translated_question,
        "retrieval_type": "tRAG",
        "hits": hits,
        "chunks": chunks,
        "retrieval_time": retrieval_time,
        "translate_time": translate_time,
        "generate_answer_time": 0.0
    }

    if generate:
        start = time.perf_counter()
        result["answer"] = generate_answer(translated_question, chunks)
        result["generate_answer_time"] = time.perf_counter() - start

    return result