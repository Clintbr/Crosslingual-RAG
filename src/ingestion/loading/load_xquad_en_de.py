"""
load_xquad_en_de.py
===================

Loads XQuAD (Cross-lingual Question Answering Dataset) for English and
German and exports it into the same format used by the original loader:

  1. Contexts  -> plain .txt files, max MAX_CHARS_PER_FILE characters each
                 (long contexts are split into part1, part2, ... files)
  2. Questions -> one JSON file per language:
                 {id, question, question_lang, answer,
                  context_lang, context}
  3. Statistics -> JSON summary per language + parallelism check

XQuAD contains 1,190 parallel QA pairs from the SQuAD v1.1 development set.
The IDs are shared across the language versions, so EN and DE can be joined
directly by id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as stats
from collections import defaultdict
from typing import Any

from src.config import BASE_PATH
from src.utils.resolve_path import resolve_project_path

HF_DATASET_NAME = "google/xquad"

XQUAD_LANGS = {"en", "de"}

OUTPUT_DIR = resolve_project_path(BASE_PATH)
CONTEXTS_DIR = OUTPUT_DIR / "raw"
QUESTIONS_DIR = OUTPUT_DIR / "x_questions"
STATS_FILE = OUTPUT_DIR / "x_load_statics/" "xquad_load_statistics.json"

DEFAULT_TARGET_LANGS = ["en", "de"]
MAX_CHARS_PER_FILE_DEFAULT = 10000


def load_language(lang: str) -> list[dict[str, Any]]:
    """Load one XQuAD language config and normalise it."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' library is not installed. Run:\n"
            "    pip install datasets"
        ) from exc

    if lang not in XQUAD_LANGS:
        raise ValueError(f"Unsupported XQuAD language: {lang}")

    print(f"Loading XQuAD {lang} from Hugging Face ...")
    ds = load_dataset(HF_DATASET_NAME, f"xquad.{lang}", split="validation")

    records: list[dict[str, Any]] = []

    for row in ds:
        answers = row.get("answers") or {}
        texts = answers.get("text") or []
        answer_text = texts[0] if texts else ""

        records.append({
            "id": row["id"],
            "question": row["question"],
            "question_lang": lang,
            "answer": answer_text,
            "context_lang": lang,
            "context": row["context"],
        })

    print(f"  loaded {len(records)} records for '{lang}'")
    return records

def context_id_for(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

def write_context_files(
        contexts_by_lang: dict[str, set[str]],
        max_chars: int
) -> None:
    for lang, contexts in contexts_by_lang.items():
        lang_dir = CONTEXTS_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        for context in contexts:
            cid = context_id_for(context)
            chunks = [
                context[i:i + max_chars]
                for i in range(0, max(len(context), 1), max_chars)
            ]

            if not chunks:
                chunks = [""]

            for part_num, chunk in enumerate(chunks, start=1):
                fname = (
                    f"{cid}.txt"
                    if len(chunks) == 1
                    else f"{cid}_part{part_num}.txt"
                )
                (lang_dir / fname).write_text(chunk, encoding="utf-8")


def write_question_files(
        records_by_lang: dict[str, list[dict[str, Any]]]
) -> None:
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for lang, records in records_by_lang.items():
        out_path = QUESTIONS_DIR / f"questions_{lang}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"  wrote {len(records):>6} questions -> {out_path}")


def build_statistics(
        target_langs: list[str],
        records_by_lang: dict[str, list[dict[str, Any]]],
        contexts_by_lang: dict[str, set[str]],
) -> dict[str, Any]:

    summary: dict[str, Any] = {
        "requested_languages": target_langs,
        "dataset": "google/xquad",
        "split": "validation",
        "per_language": {},
    }

    for lang in target_langs:
        lang_records = records_by_lang.get(lang, [])

        q_lengths = [
            len(r["question"])
            for r in lang_records
            if r["question"]
        ]
        a_lengths = [
            len(r["answer"])
            for r in lang_records
            if r["answer"]
        ]
        c_lengths = [
            len(c)
            for c in contexts_by_lang.get(lang, set())
        ]

        entry: dict[str, Any] = {
            "available_in_xquad": lang in XQUAD_LANGS,
            "num_questions_loaded": len(lang_records),
            "num_unique_contexts": len(
                contexts_by_lang.get(lang, set())
            ),
        }

        if q_lengths:
            entry["question_length_chars"] = {
                "mean": round(stats.mean(q_lengths), 1),
                "min": min(q_lengths),
                "max": max(q_lengths),
            }

        if a_lengths:
            entry["answer_length_chars"] = {
                "mean": round(stats.mean(a_lengths), 1),
                "min": min(a_lengths),
                "max": max(a_lengths),
            }

        if c_lengths:
            entry["context_length_chars"] = {
                "mean": round(stats.mean(c_lengths), 1),
                "min": min(c_lengths),
                "max": max(c_lengths),
            }

        summary["per_language"][lang] = entry

    id_sets = {
        lang: {r["id"] for r in recs}
        for lang, recs in records_by_lang.items()
        if recs
    }

    if len(id_sets) >= 2:
        common_ids = set.intersection(*id_sets.values())
        union_ids = set.union(*id_sets.values())

        summary["parallelism_check"] = {
            "languages_compared": sorted(id_sets.keys()),
            "ids_common_to_all_requested_languages": len(common_ids),
            "ids_in_union_of_all_languages": len(union_ids),
            "fully_parallel": (
                    len(common_ids)
                    == len(union_ids)
                    == max(len(s) for s in id_sets.values())
            ),
        }

    summary["total_questions_loaded"] = sum(
        len(v) for v in records_by_lang.values()
    )

    return summary


def run_loader_xquad() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--langs",
        nargs="+",
        default=DEFAULT_TARGET_LANGS,
        choices=["en", "de"],
        help="Languages to keep. Default: en de",
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records_by_lang: dict[str, list[dict[str, Any]]] = {}

    for lang in args.langs:
        records_by_lang[lang] = load_language(lang)

    contexts_by_lang: dict[str, set[str]] = defaultdict(set)

    for lang, records in records_by_lang.items():
        for record in records:
            contexts_by_lang[lang].add(record["context"])

    print(
        f"\nWriting context .txt files "
        f"(max {args.max_chars} chars each) ..."
    )
    write_context_files(contexts_by_lang, args.max_chars)

    print("\nWriting question JSON files ...")
    write_question_files(records_by_lang)

    print("\nComputing statistics ...")
    summary = build_statistics(
        args.langs,
        records_by_lang,
        contexts_by_lang
    )

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Statistics written to {STATS_FILE}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_loader_xquad()
