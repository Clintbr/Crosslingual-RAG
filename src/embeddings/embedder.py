import hashlib

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.chunking.chunker import chunk_document
from src.config import (
    QDRANT_PORT, QDRANT_HOST, EMBED_MODEL, COLLECTION_NAME, VECTOR_DIMENSION
)

q_client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
if not q_client.collection_exists(COLLECTION_NAME):
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
    )

embedder = SentenceTransformer(EMBED_MODEL)

def embedd_document(filepath):
    chunks_mongo = chunk_document(filepath)
    if not chunks_mongo:
        print("Skipping embedding...🟠")
        return
    file_name = chunks_mongo[0].get('filename')
    batch_points = []
    chunk_count = 0

    print(f"Start embedding for Chunks for: {file_name}...🟩")
    for i, chunk in enumerate(chunks_mongo):

        # Vektorisierung
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

    print(f"Embedding finished for Chunks of: {file_name} -> {chunk_count}...🟩")

    return batch_points, file_name

def upload_document(filepath):
    embedding = embedd_document(filepath)
    if not embedding:
        print("Skipping Qdrant Upload...🟠")
        return
    batch_points, file_name = embedding
    # 5. In Qdrant hochladen
    if batch_points:
        q_client.upsert(collection_name=COLLECTION_NAME, points=batch_points)

    print(f"Ingestion finished for file: {file_name} ...🟩")
