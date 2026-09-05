from nltk.corpus.reader import documents
from qdrant_client import QdrantClient
import requests
import time
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bson import ObjectId
from torch.distributed.autograd import context

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

def build_context(docs: [], translated=False) -> str:
    built_docs = docs
    for document in built_docs:
        source = f"Source:: {document['filename']} - Page {document['page']}: \n"
        if translated:
            content = source + document["translation"]
        else:
            content = source + document["content"]
        document.update({"_built_context": content})

    context = "\n\n".join(
        b.get("_built_context", "")
        for b in built_docs
    )
    return context


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
def generate_answer(question: str, context: str, target_lang: str):

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
            "stream": False,
            "options": {
                "temperature": 0.0,  # set to null to avoid high hallucination
                "num_ctx": 4096  # big window for more context
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()
