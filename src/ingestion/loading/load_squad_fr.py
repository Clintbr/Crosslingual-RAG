"""
load_squad_fr.py
================

Loads SQuAD-fr (qwant/squad_fr) and keeps ONLY the French QA records
whose IDs are present in XQuAD.

This is important for the EN-DE-FR parallel dataset:

    SQuAD/XQuAD id
        |
        +-- XQuAD English
        +-- XQuAD German
        +-- SQuAD-fr French

XQuAD contains 1,190 QA pairs from the SQuAD v1.1 development set.
SQuAD-fr is a French translation of SQuAD and preserves the SQuAD QA
IDs. Therefore, the XQuAD IDs can be used to filter the French data.

The script deliberately loads only the SQuAD-fr validation split because
XQuAD is based on the SQuAD v1.1 development/validation split.

Output structure is kept compatible with load_xquad_en_de.py:

  OUTPUT_DIR/
      CONTEXTS_DIR/
          fr/
              <hash>.txt
              ...
      QUESTIONS_DIR/
          questions_fr.json
      STATS_FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as stats
from typing import Any

from src.config import BASE_PATH
from src.utils.resolve_path import resolve_project_path

SQUAD_FR_DATASET = "qwant/squad_fr"
SQUAD_FR_SPLIT = "validation"

XQUAD_DATASET = "google/xquad"
XQUAD_CONFIG = "xquad.en"
XQUAD_SPLIT = "validation"

OUTPUT_DIR = resolve_project_path(BASE_PATH)
CONTEXTS_DIR = OUTPUT_DIR / "raw"
QUESTIONS_DIR = OUTPUT_DIR / "x_questions"
STATS_FILE = OUTPUT_DIR / "x_load_statics/squad_load_statistics.json"

MAX_CHARS_PER_FILE_DEFAULT = 10000

def load_xquad_ids(dataset_name: str) -> set[str]:
    """Load the XQuAD English IDs used as the French filter."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' library is not installed. Run:\n"
            "    pip install datasets"
        ) from exc

    print(
        f"Loading XQuAD English IDs from "
        f"{dataset_name} ..."
    )

    ds = load_dataset(
        dataset_name,
        XQUAD_CONFIG,
        split=XQUAD_SPLIT
    )

    ids = {row["id"] for row in ds}

    print(f"  XQuAD English contains {len(ids)} unique IDs")

    if not ids:
        raise RuntimeError(
            "No XQuAD IDs were loaded; refusing to create a French dataset."
        )

    return ids


def load_french_records(
        xquad_ids: set[str],
        dataset_name: str
) -> list[dict[str, Any]]:
    """
    Load SQuAD-fr without using its legacy `squad_fr.py` dataset script.

    Recent versions of `datasets` no longer execute dataset loading scripts.
    qwant/squad_fr is an older script-based dataset, so we download its
    original dummy_data.zip with huggingface_hub and parse dev-v1.1.json
    directly.

    The dataset repository documents the validation split as 17,492 records
    and exposes the fields:
        id, title, context, question, answers

    Only IDs occurring in XQuAD are retained.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "The 'huggingface_hub' library is not installed. Run:\n"
            "    pip install huggingface_hub"
        ) from exc

    import tempfile
    import zipfile

    print(
        f"Downloading French SQuAD data from {dataset_name} "
        f"(legacy dataset script bypassed) ..."
    )

    # qwant/squad_fr stores the actual source JSON files inside this archive.
    archive_path = hf_hub_download(
        repo_id=dataset_name,
        filename="dummy/1.1.0/dummy_data.zip",
        repo_type="dataset",
    )

    records: list[dict[str, Any]] = []
    matched_ids: set[str] = set()

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.namelist()

        dev_members = [
            member
            for member in members
            if member.endswith("dev-v1.1.json")
        ]

        if not dev_members:
            raise RuntimeError(
                "Could not find dev-v1.1.json inside SQuAD-fr archive. "
                f"Archive members: {members}"
            )

        dev_member = dev_members[0]

        with archive.open(dev_member) as f:
            squad = json.load(f)

    for article in squad.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "").strip()

            for qa in paragraph.get("qas", []):
                row_id = qa.get("id")

                if row_id not in xquad_ids:
                    continue

                answers = qa.get("answers") or []
                answer_text = (
                    answers[0].get("text", "").strip()
                    if answers
                    else ""
                )

                records.append({
                    "id": row_id,
                    "question": qa.get("question", "").strip(),
                    "question_lang": "fr",
                    "answer": answer_text,
                    "context_lang": "fr",
                    "context": context,
                })

                matched_ids.add(row_id)

    missing_ids = xquad_ids - matched_ids
    extra_ids = matched_ids - xquad_ids

    print(f"  matched French records: {len(records)}")
    print(f"  missing XQuAD IDs in SQuAD-fr: {len(missing_ids)}")
    print(f"  unexpected matched IDs: {len(extra_ids)}")

    if missing_ids:
        sample = sorted(missing_ids)[:10]
        raise RuntimeError(
            "SQuAD-fr does not contain every XQuAD ID. "
            f"First missing IDs: {sample}"
        )

    if extra_ids:
        raise RuntimeError(
            "Filtering error: French records contain IDs outside XQuAD."
        )

    if len(records) != len(xquad_ids):
        raise RuntimeError(
            f"Expected {len(xquad_ids)} French records, "
            f"but got {len(records)}."
        )

    return records

def context_id_for(text: str) -> str:
    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:12]


def write_context_files(
        contexts: set[str],
        max_chars: int
) -> None:

    lang_dir = CONTEXTS_DIR / "fr"
    lang_dir.mkdir(parents=True, exist_ok=True)

    for context in contexts:
        cid = context_id_for(context)

        chunks = [
            context[i:i + max_chars]
            for i in range(
                0,
                max(len(context), 1),
                max_chars
            )
        ]

        if not chunks:
            chunks = [""]

        for part_num, chunk in enumerate(chunks, start=1):
            fname = (
                f"{cid}.txt"
                if len(chunks) == 1
                else f"{cid}_part{part_num}.txt"
            )

            (lang_dir / fname).write_text(
                chunk,
                encoding="utf-8"
            )


def write_question_file(
        records: list[dict[str, Any]]
) -> None:

    QUESTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = QUESTIONS_DIR / "questions_fr.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"  wrote {len(records):>6} questions -> {out_path}"
    )


def build_statistics(
        xquad_ids: set[str],
        records: list[dict[str, Any]],
        contexts: set[str],
) -> dict[str, Any]:

    q_lengths = [
        len(r["question"])
        for r in records
        if r["question"]
    ]

    a_lengths = [
        len(r["answer"])
        for r in records
        if r["answer"]
    ]

    c_lengths = [
        len(c)
        for c in contexts
    ]

    french_ids = {
        record["id"]
        for record in records
    }

    common_ids = xquad_ids & french_ids

    summary: dict[str, Any] = {
        "requested_languages": ["fr"],
        "dataset": SQUAD_FR_DATASET,
        "split": SQUAD_FR_SPLIT,
        "source_filter": {
            "dataset": XQUAD_DATASET,
            "config": XQUAD_CONFIG,
            "split": XQUAD_SPLIT,
            "num_xquad_ids": len(xquad_ids),
        },
        "per_language": {
            "fr": {
                "available_in_squad_fr": True,
                "num_questions_loaded": len(records),
                "num_unique_contexts": len(contexts),
            }
        },
        "parallelism_check": {
            "languages_compared": ["xquad_en", "fr"],
            "xquad_ids": len(xquad_ids),
            "french_ids": len(french_ids),
            "ids_common_to_xquad_and_french": len(common_ids),
            "fully_parallel": (
                    len(common_ids) == len(xquad_ids) == len(french_ids)
            ),
        },
        "total_questions_loaded": len(records),
    }

    if q_lengths:
        summary["per_language"]["fr"]["question_length_chars"] = {
            "mean": round(stats.mean(q_lengths), 1),
            "min": min(q_lengths),
            "max": max(q_lengths),
        }

    if a_lengths:
        summary["per_language"]["fr"]["answer_length_chars"] = {
            "mean": round(stats.mean(a_lengths), 1),
            "min": min(a_lengths),
            "max": max(a_lengths),
        }

    if c_lengths:
        summary["per_language"]["fr"]["context_length_chars"] = {
            "mean": round(stats.mean(c_lengths), 1),
            "min": min(c_lengths),
            "max": max(c_lengths),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--xquad-dataset",
        default=XQUAD_DATASET,
        help="Hugging Face XQuAD dataset name",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CHARS_PER_FILE_DEFAULT,
        help="Maximum characters per context .txt file",
    )

    args = parser.parse_args()

    if args.max_chars <= 0:
        raise SystemExit("--max-chars must be greater than 0")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. Get the exact XQuAD subset IDs.
    xquad_ids = load_xquad_ids(
        args.xquad_dataset
    )

    # 2. Load French SQuAD and retain only those IDs.
    records = load_french_records(
        xquad_ids,
        SQUAD_FR_DATASET
    )

    # 3. Collect unique French contexts.
    contexts: set[str] = {
        record["context"]
        for record in records
    }

    print(
        f"\nWriting French context .txt files "
        f"(max {args.max_chars} chars each) ..."
    )

    write_context_files(
        contexts,
        args.max_chars
    )

    print("\nWriting French question JSON ...")

    write_question_file(records)

    print("\nComputing statistics ...")

    summary = build_statistics(
        xquad_ids,
        records,
        contexts
    )

    with open(
            STATS_FILE,
            "w",
            encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nDone. Statistics written to {STATS_FILE}"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
