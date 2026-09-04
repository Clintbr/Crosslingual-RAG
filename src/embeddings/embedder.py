import hashlib

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.chunking.chunker import chunk_document
from src.config import (
    QDRANT_PORT, QDRANT_HOST, EMBED_MODEL, COLLECTION_NAME
)

q_client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBED_MODEL)

if not q_client.collection_exists(COLLECTION_NAME):
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

def embedd_document(filepath):
    chunks_mongo = chunk_document(filepath)
    file_name = chunks_mongo[0].get('filename')
    batch_points = []
    chunk_count = 0

    print(f"Start embedding for Chunks aus: {file_name}...🟩")
    for i, chunk in enumerate(chunks_mongo):

        # 4. Vektorisierung
        vector = embedder.encode(chunk.get('content')).tolist()

        # Eindeutige ID für Qdrant erzeugen
        qdrant_id = hashlib.sha256(f"{chunk.get('file_hash')}_{chunk_count}".encode()).hexdigest()[:32]

        batch_points.append(
            PointStruct(
                id=qdrant_id,
                vector=vector,
                payload={
                    "mongo_id": chunk.get('_id'),
                    "filename": chunk.get('filename'),
                    "lang": chunk.get('lang'),
                    "page": chunk.get('page'),
                }
            )
        )
        chunk_count += 1

    print(f"Embedding finished for Chunks aus: {file_name} -> {chunk_count}...🟩")

    return batch_points, file_name

def upload_document(filepath):
    batch_points, file_name = embedd_document(filepath)
    # 5. In Qdrant hochladen
    if batch_points:
        q_client.upsert(collection_name=COLLECTION_NAME, points=batch_points)

    print(f"Ingestion finished for file: {file_name} ->  intelligente Chunks erstellt...🟩")
