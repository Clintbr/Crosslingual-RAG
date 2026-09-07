from qdrant_client import QdrantClient
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
)

# DB/ Clients
m_client = MongoClient(MONGO_URI)
db = m_client[DB_NAME]
chunks_col = db["chunks"]

q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

limit = 10

# Query-Embedding
_embedder = SentenceTransformer(EMBED_MODEL)

def embed_text(text: str):
    return _embedder.encode(text).tolist()

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

def build_context(docs: [], translated=False) -> str:
    built_context = []
    for document in docs:
        source = f"Source:: {document['filename']} - Page {document['page']}: \n"
        if translated:
            content = source + document["translation"]
        else:
            content = source + document["content"]
        built_context.append(content)

    context = "\n\n".join(
        b
        for b in built_context
    )
    return context
