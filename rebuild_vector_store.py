from pathlib import Path
from data_loader import load_documents
from vector_store import build_vector_store

def rebuild_vector_store():
    documents = load_documents()
    store_dir = Path("data/vector_store")
    build_vector_store(documents, store_dir=store_dir)

if __name__ == "__main__":
    rebuild_vector_store()

