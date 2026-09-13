import json
import csv


def save_hardware_metrics(input_json: str, output_csv: str) -> None:
    """
    Read a JSON file containing a list of question objects (each with an
    "id" and a "hardware" list), and write one CSV row per hardware entry.

    The "avg_vram_mb" and "peak_vram_mb" fields are intentionally skipped.
    """

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fieldnames = [
        "question_id",
        "avg_ram_mb",
        "peak_ram_mb",
        "avg_cpu_percent",
        "peak_cpu_percent",
        "rag_strategy",
        "phase",
    ]

    # resume
    completed_entries = set()

    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                key = (
                    row.get("question_id"),
                    row.get("rag_strategy"),
                    row.get("phase"),
                )
                completed_entries.add(key)

    except FileNotFoundError:
        pass

    file_exists = False

    try:
        with open(output_csv, "r", encoding="utf-8") as f:
            file_exists = f.read(1) != ""
    except FileNotFoundError:
        file_exists = False

    mode = "a" if file_exists else "w"

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        new_entries = 0
        skipped_entries = 0

        for entry in data:
            question_id = entry.get("id")

            for hw in entry.get("hardware", []):
                key = (
                    question_id,
                    hw.get("rag_strategy"),
                    hw.get("phase"),
                )

                if key in completed_entries:
                    skipped_entries += 1
                    continue

                writer.writerow({
                    "question_id": question_id,
                    "avg_ram_mb": hw.get("avg_ram_mb"),
                    "peak_ram_mb": hw.get("peak_ram_mb"),
                    "avg_cpu_percent": hw.get("avg_cpu_percent"),
                    "peak_cpu_percent": hw.get("peak_cpu_percent"),
                    "rag_strategy": hw.get("rag_strategy"),
                    "phase": hw.get("phase"),
                })

                f.flush()

                completed_entries.add(key)
                new_entries += 1

    print(
        f"Saved {new_entries} new hardware metrics "
        f"from {input_json} to {output_csv}"
    )

    if skipped_entries:
        print(
            f"Skipped {skipped_entries} already existing hardware entries."
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python save_hardware_metrics.py "
            "<input_json> <output_csv>"
        )
        sys.exit(1)

    save_hardware_metrics(
        sys.argv[1],
        sys.argv[2]
    )

    print(
        f"Saved hardware metrics from "
        f"'{sys.argv[1]}' to '{sys.argv[2]}'"
    )