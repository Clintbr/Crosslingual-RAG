import json

from src.config import QUESTIONS_DIRECTORY, PREPARED_QUESTIONS_DIRECTORY
from src.utils.resolve_path import resolve_project_path


BASE_DIR = resolve_project_path(QUESTIONS_DIRECTORY)
INPUT_FILES = {
    "de": BASE_DIR / "questions_de.json",
    "en": BASE_DIR / "questions_en.json",
    "fr": BASE_DIR / "questions_fr.json",
}
OUTPUT_BASE_DIR = resolve_project_path(PREPARED_QUESTIONS_DIRECTORY)

MONORAG_DIR = OUTPUT_BASE_DIR / "monoRAG"
MULTIRAG_DIR = OUTPUT_BASE_DIR / "multiRAG"
CROSSRAG_DIR = OUTPUT_BASE_DIR / "crossRAG"
TRAG_DIR = OUTPUT_BASE_DIR / "tRAG"

def load_questions(path):
    """Load questions from a JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_questions(questions, path):
    """Save questions to a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            questions,
            file,
            ensure_ascii=False,
            indent=2
        )


def sort_questions(questions):
    """Sort questions by ID."""

    return sorted(questions, key=lambda x: x["id"])


def create_id_mapping(questions, language):
    """
    Create a mapping:

        question_id -> question

    for one language.
    """

    mapping = {}

    for question in questions:

        question_id = question["id"]

        if question_id in mapping:
            raise ValueError(
                f"Duplicate ID '{question_id}' in {language} questions."
            )

        mapping[question_id] = question

    return mapping


def validate_ids(question_sets):
    """
    Make sure all three language datasets contain
    exactly the same question IDs.
    """

    languages = list(question_sets.keys())

    reference_ids = {
        question["id"]
        for question in question_sets[languages[0]]
    }

    for language in languages[1:]:

        current_ids = {
            question["id"]
            for question in question_sets[language]
        }

        missing = reference_ids - current_ids
        extra = current_ids - reference_ids

        if missing:
            raise ValueError(
                f"IDs missing in {language}: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Extra IDs in {language}: {sorted(extra)}"
            )

def prepare_monorag(question_sets):

    print("Preparing MonoRAG...")

    for language, questions in question_sets.items():

        sorted_questions = sort_questions(questions)

        output_path = MONORAG_DIR / f"questions_{language}.json"

        save_questions(
            sorted_questions,
            output_path
        )

        print(f"  Saved: {output_path}")

def prepare_multirag(question_sets):

    print("Preparing MultiRAG...")

    # Create mappings for accessing the same question ID
    mappings = {
        language: create_id_mapping(
            questions,
            language
        )
        for language, questions in question_sets.items()
    }

    # Use sorted IDs from the German dataset as reference
    sorted_ids = sorted(mappings["de"].keys())

    for question_language in ["de", "en", "fr"]:

        result = []

        for question_id in sorted_ids:

            question = mappings[question_language][question_id]

            context_de = mappings["de"][question_id]["context"]
            context_en = mappings["en"][question_id]["context"]
            context_fr = mappings["fr"][question_id]["context"]

            new_question = question.copy()

            new_question["context_lang"] = "de-en-fr"

            new_question["context"] = (
                    context_de
                    + "\n"
                    + context_en
                    + "\n"
                    + context_fr
            )

            result.append(new_question)

        output_path_1 = MULTIRAG_DIR / f"questions_{question_language}_1.json"
        output_path_2 = MULTIRAG_DIR / f"questions_{question_language}_2.json"

        mitte = len(result) // 2
        save_questions(
            result[:mitte],
            output_path_1
        )
        save_questions(
            result[mitte:],
            output_path_2
        )

        print(f"  Saved: {output_path_1} \n Saved: {output_path_2}")

def prepare_crossrag(question_sets):

    print("Preparing CrossRAG...")

    mappings = {
        language: create_id_mapping(
            questions,
            language
        )
        for language, questions in question_sets.items()
    }

    sorted_ids = sorted(mappings["de"].keys())

    for question_language in ["de", "en", "fr"]:

        result = []

        for question_id in sorted_ids:

            question = mappings[question_language][question_id]

            # Context in the language of the question
            context = mappings[question_language][question_id]["context"]

            new_question = question.copy()

            new_question["context_lang"] = "de-en-fr"
            new_question["context"] = context

            result.append(new_question)

        output_path = CROSSRAG_DIR / f"questions_{question_language}.json"

        save_questions(
            result,
            output_path
        )

        print(f"  Saved: {output_path}")

def prepare_trag(question_sets):

    print("Preparing tRAG...")

    mappings = {
        language: create_id_mapping(
            questions,
            language
        )
        for language, questions in question_sets.items()
    }

    sorted_ids = sorted(mappings["de"].keys())

    def create_dataset(
            question_language,
            context_language
    ):

        result = []

        for question_id in sorted_ids:

            question = mappings[question_language][question_id]
            context_question = mappings[context_language][question_id]

            new_question = question.copy()

            # Answer comes from the question language
            new_question["answer"] = question["answer"]

            # Context comes from another language
            new_question["context_lang"] = context_language
            new_question["context"] = context_question["context"]

            result.append(new_question)

        return result

    dataset_fr_en = create_dataset("fr", "en")

    save_questions(
        dataset_fr_en,
        TRAG_DIR / "questions_fr_context_en.json"
    )

    dataset_fr_de = create_dataset("fr", "de")

    save_questions(
        dataset_fr_de,
        TRAG_DIR / "questions_fr_context_de.json"
    )

    dataset_en_de = create_dataset("en", "de")

    save_questions(
        dataset_en_de,
        TRAG_DIR / "questions_en_context_de.json"
    )

    dataset_en_fr = create_dataset("en", "fr")

    save_questions(
        dataset_en_fr,
        TRAG_DIR / "questions_en_context_fr.json"
    )

    dataset_de_en = create_dataset("de", "en")

    save_questions(
        dataset_de_en,
        TRAG_DIR / "questions_de_context_en.json"
    )

    dataset_de_fr = create_dataset("de", "fr")

    save_questions(
        dataset_de_fr,
        TRAG_DIR / "questions_de_context_fr.json"
    )

    print(f"  Saved 6 tRAG datasets in: {TRAG_DIR}")

def prepare_questions_for_evaluation():

    print("Loading question datasets...")

    question_sets = {
        language: load_questions(path)
        for language, path in INPUT_FILES.items()
    }

    print(
        f"Loaded: "
        f"DE={len(question_sets['de'])}, "
        f"EN={len(question_sets['en'])}, "
        f"FR={len(question_sets['fr'])}"
    )

    # Make sure all datasets contain the same IDs
    validate_ids(question_sets)

    print("All question IDs are consistent.\n")

    # Prepare datasets
    prepare_monorag(question_sets)
    prepare_multirag(question_sets)
    prepare_crossrag(question_sets)
    prepare_trag(question_sets)

    print("\nDone.")


if __name__ == "__main__":
    prepare_questions_for_evaluation()