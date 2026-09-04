import os
import hashlib
from pymongo import MongoClient
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract
from datetime import datetime
from src.config import (
    MONGO_URI, DB_NAME, TESSERACT_PATH, TESSERACT_EXEC_PATH
)

tesseract_exec_path = TESSERACT_EXEC_PATH

if os.path.exists(tesseract_exec_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_exec_path
    os.environ["PATH"] += os.pathsep + TESSERACT_PATH
else:
    print(f"WARNUNG: Tesseract wurde unter {tesseract_exec_path} nicht gefunden!")

m_client = MongoClient(MONGO_URI)
db = m_client[DB_NAME]
chunks_col = db["chunks"]
files_col = db["processed_files"]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=160,
    separators=["\n\n", "\n", ".", " ", ""]
)

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def chunk_document(filepath):
    filename = os.path.basename(filepath)
    file_hash = get_file_hash(filepath)
    chunk_count = 0
    chunks_mongo = []

    print(f"Start Ingestion for: {filename}...🟩")
    if files_col.find_one({"file_hash": file_hash}):
        print(f"Skipping process for {filename}: already ingested.🟠")
        print("Skipping chunking...🟠")
        return
    print(f"Extract text from: {filename}...🟩")

    # 1. text extraction with unstructured
    elements = partition(
        filename=filepath,
        strategy="hi_res" if filepath.endswith(".pdf") else "auto"
    )

    # 2. we group the text by pagenumber to allow a page based chunking
    print(f"Start chunking for: {filename}...🟩")
    pages = {}
    for el in elements:
        p = el.metadata.to_dict().get("page_number", 1)
        pages[p] = pages.get(p, "") + "\n" + str(el)

    # 3. Intelligent page-level chunking
    for page_num, page_text in pages.items():
        if len(page_text.strip()) < 10: continue

        # the splitter pays attention to paragraphs and sentences.
        chunks = text_splitter.split_text(page_text)

        for i, chunk_content in enumerate(chunks):
            # 3. save the chunks in MongoDB
            mongo_doc = {
                "filename": filename,
                "file_hash": file_hash,
                "content": chunk_content,
                "lang": filename[:2],
                "page": page_num,
                "chunk_index": chunk_count
            }
            m_id = str(chunks_col.insert_one(mongo_doc).inserted_id)
            mongo_doc.update({"_id": m_id})
            chunks_mongo.append(mongo_doc)
            chunk_count += 1

    # 4. save the file in MongoDB
    files_col.insert_one({
        "filename": filename,
        "file_hash": file_hash,
        "lang": filename[:2],
        "processed_at": datetime.now()
    })

    print(f"chunks saved to mongodb: {filename} -> {chunk_count} 🟩")

    return chunks_mongo