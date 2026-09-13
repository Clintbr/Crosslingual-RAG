"""
    Run the RAG methods and save the results in provided .json file
    This Code is a Checkpoint for every cross-lingual rag strategies. While saving the questions and answer
    with others relevant metrics, it avoids repeating the process of running the rag pipeline, when trying to evaluate
    many times the same entity.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.evaluating.hardware.evaluate_hardware_consume import measure_operation
from src.retrieval.cross_rag.run_cross_rag import run_cross_retrieval
from src.retrieval.mono_rag.run_mono_rag import run_mono_retrieval
from src.retrieval.multi_rag.run_multi_rag import run_multi_retrieval
from src.retrieval.t_rag.run_t_rag import run_trag_retrieval


def run_rag_method(method_name: str, question_data: dict) -> tuple[dict, Any]:
    question = question_data["question"]
    question_lang = question_data["question_lang"]

    if method_name == "mono":
        generate_result, generate_metrics = measure_operation(
            lambda: run_mono_retrieval(
                question,
                question_lang,
                generate=True,
            )
        )
        return generate_result, generate_metrics

    if method_name == "multi":
        generate_result, generate_metrics = measure_operation(
            lambda: run_multi_retrieval(
                question,
                question_lang,
                generate=True,
            )
        )
        return generate_result, generate_metrics


    if method_name == "cross":
        generate_result, generate_metrics = measure_operation(
            lambda: run_cross_retrieval(
                question,
                question_lang,
                generate=True,
            )
        )
        return generate_result, generate_metrics

    if method_name == "trag":
        generate_result, generate_metrics = measure_operation(
            lambda: run_trag_retrieval(
                question,
                question_lang,
                question_data["context_lang"],
                generate=True,
            )
        )

        return generate_result, generate_metrics

    raise ValueError(f"Unknown RAG method: {method_name}")

def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of questions.")

    return data


def load_existing_results(path: Path) -> dict[str, dict]:
    """
    Returns:
        {
            question_id: result
        }
    """

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return {
                str(item["id"]): item
                for item in data
                if item.get("id") is not None
            }

        return {}

    except (json.JSONDecodeError, OSError):
        print(f"Warning: Could not read existing results: {path}")
        return {}

def make_json_serializable(value):
    """
    Recursively convert objects into JSON-serializable values.

    Handles:
    - dict
    - list / tuple
    - primitive values
    - Pydantic models
    - objects with __dict__
    - Qdrant ScoredPoint
    """

    # Primitive values
    if value is None or isinstance(
            value,
            (str, int, float, bool),
    ):
        return value

    # Dictionaries
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(val)
            for key, val in value.items()
        }

    # Lists / tuples / sets
    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    # Pydantic models
    if hasattr(value, "model_dump"):
        return make_json_serializable(
            value.model_dump()
        )

    # Older Pydantic versions
    if hasattr(value, "dict") and callable(value.dict):
        return make_json_serializable(
            value.dict()
        )

    # Generic Python objects
    if hasattr(value, "__dict__"):
        return make_json_serializable(
            vars(value)
        )

    # Last resort
    return str(value)


def save_results(
        path: Path,
        results: dict[str, dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = make_json_serializable(
        list(results.values())
    )

    with path.open(
            "w",
            encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

def run_pipeline(
        method_name: str,
        input_json: str,
        output_json: str,
        resume: bool = True,
) -> None:

    input_path = Path(input_json)
    output_path = Path(output_json)

    loaded_questions = load_questions(input_path)
    questions = loaded_questions[:100]

    existing_results = (
        load_existing_results(output_path)
        if resume
        else {}
    )

    print("=" * 70)
    print(f"RAG PIPELINE: {method_name}")
    print(f"Input:     {input_path}")
    print(f"Output:    {output_path}")
    print(f"Questions: {len(questions)}")
    print(f"Resume:    {resume}")
    print("=" * 70)

    for index, question_data in enumerate(questions, start=1):

        question_id = str(question_data["id"])

        if resume and question_id in existing_results:
            print(
                f"[{index}/{len(questions)}] "
                f"{question_data['question']}"
            )
            print("    -> already completed, skipping")
            continue

        print(
            f"\n[{index}/{len(questions)}] "
            f"{question_data['question']}"
        )

        start = time.perf_counter()

        try:
            generate_t_results, generate_t_metrics = run_rag_method(
                method_name=method_name,
                question_data=question_data,
            )
            result, hardware = generate_t_results

            generate_t_metrics.update({"rag_strategy": "monoRAG"})
            generate_t_metrics.update({"phase": "total"})
            hardware.append(generate_t_metrics)

            execution_time = time.perf_counter() - start

            # Keep the original question metadata together
            # with the RAG result.
            stored_result = {
                "id": question_data.get("id"),
                "question": question_data.get("question"),
                "question_lang": question_data.get("question_lang"),
                "answer": question_data.get("answer"),
                "context_lang": question_data.get("context_lang"),
                "context": question_data.get("context"),

                # RAG output
                "rag_result": result,

                # Hardware information if available
                "hardware": hardware,

                # Independent measurement
                "pipeline_execution_time": execution_time,

                "pipeline_error": None,
            }

            existing_results[question_id] = stored_result

            print(
                f"    RAG execution: "
                f"{execution_time:.2f}s"
            )

        except Exception as e:

            execution_time = time.perf_counter() - start

            stored_result = {
                "id": question_data.get("id"),
                "question": question_data.get("question"),
                "question_lang": question_data.get("question_lang"),
                "answer": question_data.get("answer"),
                "context_lang": question_data.get("context_lang"),
                "context": question_data.get("context"),

                "rag_result": None,
                "hardware": None,

                "pipeline_execution_time": execution_time,

                "pipeline_error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }

            existing_results[question_id] = stored_result

            print(
                f"    ERROR: {type(e).__name__}: {e}"
            )

        # Save after every question.
        save_results(output_path, existing_results)

        print(f"    Saved -> {output_path}")

    print("\n" + "=" * 70)
    print("RAG PIPELINE FINISHED")
    print(f"Results: {output_path}")
    print("=" * 70)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run a RAG pipeline and save raw results."
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=["mono", "multi", "cross", "trag"],
        help="RAG strategy to execute.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the questions JSON file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the raw results JSON file.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results and start from the beginning.",
    )

    args = parser.parse_args()

    run_pipeline(
        method_name=args.method,
        input_json=args.input,
        output_json=args.output,
        resume=not args.no_resume,
    )
