import time

from src.config import RETRIEVAL_PHASES


# build the result when th error occurred during retrieving documents from the DBs
def build_retrieval_result_by_first_level_error(question, retrieval_time, retrieval_type, hits, hardware, t0, occurred_error_type, query_translation_time = 0.0):
    error = None
    if occurred_error_type == 1:
        error = {
            "phase": RETRIEVAL_PHASES[4],
            "type": "VectorsMissing",
            "message": "No Matching Vector Found.",
            "status_code": None
        }
    elif occurred_error_type == 2:
        error = {
            "phase": RETRIEVAL_PHASES[4],
            "type": "DocumentsMissing",
            "message": "No Matching Chunks Found.",
            "status_code": None
        }
    else:
        error = {
            "phase": RETRIEVAL_PHASES[4],
            "type": "BadEntry",
            "message": "The occurred_error_type should be either 1 or 2.",
            "status_code": None
        }

    return {
        "question": question,
        "translated_question": "",
        "answer": "",
        "retrieval_type": retrieval_type,
        "error": error,
        "hits": [],
        "chunks": [],
        "translated_documents": [],
        "context": [],
        "retrieval_time": retrieval_time,
        "generate_answer_time": 0.0,
        "translate_time": query_translation_time,
        "total_time": time.perf_counter() - t0
    }, hardware