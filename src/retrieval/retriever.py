from qdrant_client import QdrantClient
import requests
import time
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bson import ObjectId

from src.config import (
    MONGO_URI,
    DB_NAME,
    COLLECTION_NAME,
    EMBED_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    OLLAMA_URL,
    TRANSLATION_MODEL,
    GENERATION_MODEL
)

# DB/ Clients
m_client = MongoClient(MONGO_URI)
db = m_client[DB_NAME]
chunks_col = db["chunks"]

q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

limit = 10

# Mongo Fetch
def fetch_chunks_from_mongo(hits):
    mongo_ids = []

    for hit in hits:
        mongo_id = hit.payload.get("mongo_id")
        if mongo_id:
            mongo_ids.append(ObjectId(mongo_id))

    if not mongo_ids:
        return []

    chunks = chunks_col.find({"_id": {"$in": mongo_ids}})

    chunk_map = {str(c["_id"]): c for c in chunks}

    ordered = []
    for hit in hits:
        mid = hit.payload.get("mongo_id")
        if mid in chunk_map:
            ordered.append(chunk_map[mid])

    return ordered

# Embedding
_embedder = SentenceTransformer(EMBED_MODEL)

def embed_text(text: str):
    return _embedder.encode(text).tolist()

# Query - Translation
def translate_question(question: str, target_lang: str):

    start = time.perf_counter()

    prompt = f"""
        Translate the following question to {target_lang}.
        Return ONLY the translated question.
        
        Question:
        {question}
    """

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": TRANSLATION_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    translation = response.json().get("response", "").strip()

    return translation, time.perf_counter() - start

# Document - Translation
def translate_retrieved_doc(documents: [], target_lang: str):
    translated_documents = []
    start = time.perf_counter()

    for document in documents:
        prompt = f"""
            Translate the following document to {target_lang}.
            Return ONLY the translated question.
            
            Question:
            {document}
        """
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": TRANSLATION_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

        response.raise_for_status()

        translation = response.json().get("response", "").strip()
        translated_documents.append(translation)

    return translated_documents, time.perf_counter() - start

# Qdrant Search
def search_qdrant(vector, lang=None):
    query_kwargs = {
        "collection_name": COLLECTION_NAME,
        "query": vector,
        "limit": limit,
        "with_payload": True
    }

    if lang:
        query_kwargs["query_filter"] = {
            "must": [
                {
                    "key": "lang",
                    "match": {"value": lang}
                }
            ]
        }

    return q_client.query_points(**query_kwargs).points


# Answer Generation
def generate_answer(question: str, chunks):

    context = "\n\n".join(
        c.get("content", "") if isinstance(c, dict) else str(c)
        for c in chunks
    )

    prompt = f"""
        You are a QA system.
        Answer ONLY using context.
        Give only the Answer back with no explanations.
        The answer should be in  same language as the question 
        If not found: "I don't know."
        
        Question:
        {question}
        
        Context:
        {context}
        
        Answer:
    """

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["response"].strip()
