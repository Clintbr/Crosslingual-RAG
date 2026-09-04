import os
from tqdm import tqdm

from src.config import DOCS_PATH
from src.embeddings.embedder import upload_document
from src.utils.resolve_path import resolve_project_path


def start_ingestion():
    print(f"Start Ingestion for every files ...🟩")
    resolve_project_path(DOCS_PATH)
    path = resolve_project_path(DOCS_PATH)

    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        return

    files = [f for f in os.listdir(path) if f.endswith(('.pdf', '.txt', '.md'))]
    for f in tqdm(files, desc="Ingestion"):
        print('processing', f, '\n')
        upload_document(os.path.join(path, f))
    print(f"Ingestion finished for every files ...🟩🟩🟩")


if __name__ == "__main__":
    start_ingestion()
