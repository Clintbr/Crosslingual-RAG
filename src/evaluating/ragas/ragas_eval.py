"""
The 6 metrics of RAGAS process.

Resume support:
- Already evaluated samples are detected by their "id" in the output CSV.
- When the script is restarted, already processed questions are skipped.
- New results are appended to the existing CSV instead of overwriting it.

Usage:
    python ragas_eval.py "input.json" "output.csv"

Metrics computed (same definitions/formulas RAGAS uses):
    faithfulness        - fraction of response claims supported by retrieved context
    answer_relevancy    - mean cosine similarity between hypothetical questions
                          (generated from the response) and the actual question
    answer_correctness  - weighted blend of factual F1 (response vs reference claims)
                          and semantic similarity (response vs reference embedding)
    context_precision   - average precision of retrieved contexts, ranked, judged
                          against the reference answer
    context_recall      - fraction of reference claims supported by retrieved context
    context_relevance   - fraction of retrieved contexts judged relevant to the question

How it stays fast:
    Instead of RAGAS running each metric as its own multi-call LLM pipeline
    (faithfulness alone issues one call per claim, context precision re-judges
    each chunk separately, etc.), this script asks the judge LLM for ALL the
    raw judgments needed for ALL six metrics in a SINGLE structured JSON
    response, then computes every metric deterministically in Python from
    that one response. A second call batches all embeddings needed
    (question, hypothetical questions, response, reference) in one request.
"""

import csv
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

from src.config import OLLAMA_BASE_URL, EVALUATION_MODEL, EMBEDDING_MODEL
from src.prompts.prompt_eval_templates import JUDGE_PROMPT_TEMPLATE


OLLAMA_HOST = OLLAMA_BASE_URL
JUDGE_MODEL = EVALUATION_MODEL
NUM_CTX = 8192
REQUEST_TIMEOUT = 300    # seconds
MAX_RETRIES = 2

# Default RAGAS weighting for answer_correctness:
# factual overlap vs semantic similarity
CORRECTNESS_WEIGHTS = (0.75, 0.25)


def _post(path: str, payload: dict) -> dict:
    url = f"{OLLAMA_HOST}{path}"
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_chat_json(prompt: str) -> dict:
    """Call Ollama chat in forced-JSON mode and return the parsed object."""

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX
        }
    }

    last_err = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = _post("/api/chat", payload)
            raw = result["message"]["content"]

            return _parse_json_loose(raw)

        except Exception as e:  # noqa: BLE001
            last_err = e

            print(
                f"  ! Judge attempt {attempt + 1}/{MAX_RETRIES + 1} failed: {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(1)

    raise RuntimeError(
        f"Ollama judge call failed after retries: {last_err}"
    )


def _parse_json_loose(raw: str) -> dict:
    """Strip markdown fences etc. before parsing."""

    text = raw.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def ollama_embed(texts: list[str]) -> list[list[float]]:
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    result = _post("/api/embed", payload)

    return result["embeddings"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

def build_prompt(
        question: str,
        response: str,
        reference: str,
        contexts: list[str]
) -> str:

    numbered_contexts = "\n".join(
        f"[{i}] {c}"
        for i, c in enumerate(contexts)
    )

    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        contexts=numbered_contexts if numbered_contexts else "(none retrieved)",
        response=response,
        reference=reference,
    )


def _safe_ratio(
        numerator: int,
        denominator: int,
        default: float = 1.0
) -> float:

    if denominator == 0:
        return default

    return numerator / denominator


def compute_faithfulness(judgment: dict) -> float:

    claims = judgment.get("response_claims", [])

    if not claims:
        return 1.0

    supported = sum(
        1
        for c in claims
        if c.get("supported_by_context")
    )

    return _safe_ratio(
        supported,
        len(claims)
    )


def compute_context_recall(judgment: dict) -> float:

    claims = judgment.get("reference_claims", [])

    if not claims:
        return 1.0

    supported = sum(
        1
        for c in claims
        if c.get("supported_by_context")
    )

    return _safe_ratio(
        supported,
        len(claims)
    )


def compute_context_relevance(judgment: dict) -> float:

    ctx = judgment.get("context_judgments", [])

    if not ctx:
        return 0.0

    relevant = sum(
        1
        for c in ctx
        if c.get("relevant_to_question")
    )

    return relevant / len(ctx)


def compute_context_precision(judgment: dict) -> float:
    """Average Precision over ranked contexts."""

    ctx = sorted(
        judgment.get("context_judgments", []),
        key=lambda c: c.get("index", 0)
    )

    if not ctx:
        return 0.0

    relevant_flags = [
        1 if c.get("useful_for_reference_answer") else 0
        for c in ctx
    ]

    total_relevant = sum(relevant_flags)

    if total_relevant == 0:
        return 0.0

    precisions = []
    hits = 0

    for k, rel in enumerate(relevant_flags, start=1):

        if rel:
            hits += 1
            precisions.append(hits / k)

    return sum(precisions) / total_relevant


def compute_correctness_f1(judgment: dict) -> float:

    comp = judgment.get(
        "correctness_comparison",
        {}
    )

    tp = len(
        comp.get("true_positives", [])
    )

    fp = len(
        comp.get("false_positives", [])
    )

    fn = len(
        comp.get("false_negatives", [])
    )

    denom = tp + 0.5 * (fp + fn)

    if denom == 0:
        return 1.0

    return tp / denom


def compute_answer_relevancy(
        question: str,
        hypothetical_questions: list[str],
        embeddings_cache: dict
) -> float:

    if not hypothetical_questions:
        return 0.0

    q_vec = embeddings_cache[question]

    sims = [
        cosine_similarity(
            q_vec,
            embeddings_cache[hq]
        )
        for hq in hypothetical_questions
    ]

    return sum(sims) / len(sims)


def compute_answer_correctness(
        response: str,
        reference: str,
        judgment: dict,
        embeddings_cache: dict
) -> float:

    factual_f1 = compute_correctness_f1(judgment)

    semantic_sim = cosine_similarity(
        embeddings_cache[response],
        embeddings_cache[reference]
    )

    w_factual, w_semantic = CORRECTNESS_WEIGHTS

    return (
            w_factual * factual_f1
            + w_semantic * semantic_sim
    )

def evaluate_row(
        question: str,
        response: str,
        reference: str,
        contexts: list[str]
) -> dict:

    prompt = build_prompt(
        question,
        response,
        reference,
        contexts
    )

    judgment = ollama_chat_json(prompt)

    hypothetical_questions = (
        judgment.get("hypothetical_questions", [])[:3]
    )

    texts_to_embed = [
                         question,
                         response,
                         reference
                     ] + hypothetical_questions

    # Remove duplicates while keeping order
    texts_to_embed = list(
        dict.fromkeys(texts_to_embed)
    )

    vectors = ollama_embed(texts_to_embed)

    embeddings_cache = dict(
        zip(texts_to_embed, vectors)
    )

    return {
        "faithfulness": compute_faithfulness(
            judgment
        ),

        "answer_relevancy": compute_answer_relevancy(
            question,
            hypothetical_questions,
            embeddings_cache
        ),

        "answer_correctness": compute_answer_correctness(
            response,
            reference,
            judgment,
            embeddings_cache
        ),

        "context_precision": compute_context_precision(
            judgment
        ),

        "context_recall": compute_context_recall(
            judgment
        ),

        "context_relevance": compute_context_relevance(
            judgment
        ),
    }

def _load_rows(input_json: str) -> list[dict]:

    with open(
            input_json,
            "r",
            encoding="utf-8"
    ) as f:

        raw = json.load(f)

    rows = []

    for item in raw:

        error = item["rag_result"]["error"]

        if error is None or len(error) == 0:

            rows.append(
                {
                    "id": item["id"],
                    "user_input": item["question"],
                    "reference": item["answer"],
                    "response": item["rag_result"]["answer"],
                    "retrieved_contexts": item["rag_result"]["context"],

                    "error_phase": None,
                    "error_type": None,
                    "error_message": None,
                    "error_status_code": None,
                }
            )

        else:

            rows.append(
                {
                    "id": item["id"],
                    "user_input": item["question"],
                    "reference": item["answer"],
                    "response": item["rag_result"]["answer"],
                    "retrieved_contexts": item["rag_result"]["context"],

                    "error_phase": error["phase"],
                    "error_type": error["type"],
                    "error_message": error["message"],
                    "error_status_code": error["status_code"],
                }
            )

    return rows

# Resume functionality
def _load_completed_ids(output_csv: str) -> set[str]:
    """
    Read already evaluated IDs from an existing output CSV.

    If the CSV does not exist or is empty, return an empty set.
    """

    output_path = Path(output_csv)

    if not output_path.exists():
        return set()

    completed_ids = set()

    try:

        with open(
                output_path,
                "r",
                newline="",
                encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                sample_id = row.get("id")

                if sample_id:
                    completed_ids.add(sample_id)

    except Exception as e:

        print(
            f"! Could not read existing output CSV: {e}"
        )

    return completed_ids

# Main evaluation pipeline
def evaluate_ragas_pipeline(
        input_json: str,
        output_csv: str
) -> None:

    rows = _load_rows(input_json)

    print(
        f"Loaded {len(rows)} samples from {input_json}"
    )

    completed_ids = _load_completed_ids(
        output_csv
    )

    if completed_ids:

        print(
            f"Resume mode: found {len(completed_ids)} "
            f"already evaluated samples."
        )

    remaining_rows = [
        row
        for row in rows
        if row["id"] not in completed_ids
    ]

    print(
        f"Remaining samples: "
        f"{len(remaining_rows)}/{len(rows)}"
    )

    if not remaining_rows:

        print(
            "All samples have already been evaluated."
        )

        return

    fieldnames = [
        "id",

        "user_input",
        "reference",
        "response",
        "retrieved_contexts",

        "error_phase",
        "error_type",
        "error_message",
        "error_status_code",

        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
        "context_relevance",
    ]

    output_path = Path(output_csv)

    # Create parent directory if necessary
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
            output_path.exists()
            and output_path.stat().st_size > 0
    )

    mode = "a" if file_exists else "w"

    with open(
            output_path,
            mode,
            newline="",
            encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        # Only write the header when creating a new file
        if not file_exists:
            writer.writeheader()

        results = []

        for i, row in enumerate(
                remaining_rows,
                start=1
        ):

            print(
                f"[{i}/{len(remaining_rows)}] "
                f"Scoring: {row['user_input'][:60]}..."
            )

            t0 = time.time()

            try:

                scores = evaluate_row(
                    row["user_input"],
                    row["response"],
                    row["reference"],
                    row["retrieved_contexts"]
                )

                success = True

            except Exception as e:  # noqa: BLE001

                print(
                    f"  ! Failed on this row: {e}"
                )

                scores = {
                    k: None
                    for k in [
                        "faithfulness",
                        "answer_relevancy",
                        "answer_correctness",
                        "context_precision",
                        "context_recall",
                        "context_relevance",
                    ]
                }

                success = False

            elapsed = time.time() - t0

            print(
                f"  done in {elapsed:.1f}s -> {scores}"
            )

            out_row = {
                "id": row["id"],

                "user_input": row["user_input"],
                "reference": row["reference"],
                "response": row["response"],

                "retrieved_contexts": json.dumps(
                    row["retrieved_contexts"],
                    ensure_ascii=False
                ),

                "error_phase": row["error_phase"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
                "error_status_code": row["error_status_code"],

                **scores,
            }

            writer.writerow(out_row)

            # Make sure the result is physically written
            # immediately. This is important for resume support
            # if the process crashes later.
            f.flush()

            results.append(scores)

            status = "OK" if success else "FAILED"

            print(
                f"  [{status}] "
                f"Saved {row['id']} to {output_csv}"
            )

    print(
        f"\nAdded {len(results)} new rows to "
        f"{output_csv}"
    )

    total_completed = len(completed_ids) + len(results)

    print(
        f"Total evaluated samples: "
        f"{total_completed}/{len(rows)}"
    )

    valid = [
        r
        for r in results
        if r["faithfulness"] is not None
    ]

    if valid:

        print("\n=== Averages (current run) ===")

        for metric in [
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "context_precision",
            "context_recall",
            "context_relevance",
        ]:

            avg = (
                    sum(r[metric] for r in valid)
                    / len(valid)
            )

            print(
                f"{metric:<20} {avg:.3f}"
            )

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 3:

        print(
            'Usage: python ragas_eval.py '
            '"input.json" "output.csv"'
        )

        sys.exit(1)

    evaluate_ragas_pipeline(
        input_json=sys.argv[1],
        output_csv=sys.argv[2]
    )