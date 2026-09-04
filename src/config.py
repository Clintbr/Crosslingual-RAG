## doc db
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "rag_eval_test"

## vector db
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "rag_eval_test_vectors"

## llm
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = ""
TRANSLATION_MODEL = "llama3"
GENERATION_MODEL = "llama3"

## languages
EN = "en"
DE = "de"
FR = "fr"

## Path
# Ingestion - convert_to_pdf
BASE_PATH = ""
DESTINATION_PATH = ""
#
PDF_DIRECTORY = ""
DOCS_PATH = "src/ingestion/test"
TESSERACT_EXEC_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
