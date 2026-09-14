## doc db

# MONGO_URI = ""
# DB_NAME = ""

## vector db
# QDRANT_HOST = ""
# QDRANT_PORT = ""
# COLLECTION_NAME = ""
# VECTOR_DIMENSION =
## llm
# OLLAMA_BASE_URL = ""
# OLLAMA_URL = "" # {OLLAMA_BASE_URL}/api/generate
# TRANSLATION_MODEL = ""
# GENERATION_MODEL = ""
# EVALUATION_MODEL = ""
# EMBED_MODEL = ""
# EMBEDDING_MODEL = ""

## languages
#EN = "en"
#DE = "de"
#FR = "fr"

## retrieval pipeline phases
# RETRIEVAL_PHASES = ["q_translation", "doc_translation", "r_translate", "generation", "mongo_retrieve"]


### Path
## Ingestion - convert_to_pdf
# BASE_PATH = "src/datasets"
#
# DOCS_PATH = "src/datasets/converted"
# TESSERACT_EXEC_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
#
# QUESTIONS_DIRECTORY = "src/datasets/x_questions"
# PREPARED_QUESTIONS_DIRECTORY = "src/evaluating/prepared_questions"
#
# retrieval_quality
# RETRIEVAL_QUALITY_RESULTS_DIRECTORY = "src/evaluating/retrieval_quality_results"
# RETRIEVAL_HARDWARE_RESULTS_DIRECTORY = "src/evaluating/hardware_consume_results"
#
# analyse results
# ANALYSIS_RESULTS_DIRECTORY = "src/evaluating/analysis_results"