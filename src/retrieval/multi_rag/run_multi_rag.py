import time

from src.retrieval.retriever import embed_text, search_qdrant, fetch_chunks_from_mongo, generate_answer


def run_multi_retrieval(question: str, generate=False):

    t0 = time.perf_counter()

    vector = embed_text(question)
    hits = search_qdrant(vector)
    chunks = fetch_chunks_from_mongo(hits)

    retrieval_time = time.perf_counter() - t0

    result = {
        "question": question,
        "retrieval_type": "crossRAG",
        "hits": hits,
        "chunks": chunks,
        "retrieval_time": retrieval_time,
        "generate_answer_time": 0.0,
        "translate_time": 0.0
    }

    if generate:
        start = time.perf_counter()
        result["answer"] = generate_answer(question, chunks)
        result["generate_answer_time"] = time.perf_counter() - start

    return result

