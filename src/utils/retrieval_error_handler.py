from src.config import RETRIEVAL_PHASES


# build the result when th error occurred during retrieving documents from the DBs
def build_retrieval_result_by_first_level_error(question, retrieval_time, retrieval_type, hits,occurred_error_type):
    if occurred_error_type is 1:
        error = {
            "phase": RETRIEVAL_PHASES[4],
            "type": "VectorsMissing",
            "message": "No Matching Vector Found.",
            "status_code": None
        }

        return {
            "question": question,
            "answer": "",
            "retrieval_type": retrieval_type,
            "error": error,
            "hits": [],
            "chunks": [],
            "retrieval_time": retrieval_time,
            "generate_answer_time": 0.0,
            "translate_time": 0.0
        }

    if occurred_error_type is 2:
        error = {
            "phase": RETRIEVAL_PHASES[4],
            "type": "DocumentsMissing",
            "message": "No Matching Chunks Found.",
            "status_code": None
        }

        return {
            "question": question,
            "answer": "",
            "retrieval_type": retrieval_type,
            "error": error,
            "hits": hits,
            "chunks": [],
            "retrieval_time": retrieval_time,
            "generate_answer_time": 0.0,
            "translate_time": 0.0
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
            "answer": "",
            "retrieval_type": "multiRAG",
            "error": error,
            "hits": hits,
            "chunks": [],
            "retrieval_time": retrieval_time,
            "generate_answer_time": 0.0,
            "translate_time": 0.0
        }
