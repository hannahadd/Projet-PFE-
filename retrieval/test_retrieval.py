import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "merged_dataset.json"
INDEX_PATH = "faiss_index.bin"
META_PATH = "metadata.json"
MODEL_NAME = "intfloat/e5-base-v2"
CHUNK_SIZE = 350  # nombre de mots par chunk
TOP_K = 5

# ==========================================
# UTILS
# ==========================================

def chunk_text(text, chunk_size=350):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# INDEX BUILDING
# ==========================================

def build_index(data):

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Chunking documents...")
    documents = []
    metadata = []

    for article in tqdm(data):
        content = article.get("content", "")
        if not content:
            continue

        chunks = chunk_text(content, CHUNK_SIZE)

        for chunk in chunks:
            documents.append(chunk)
            metadata.append({
                "title": article.get("title"),
                "source": article.get("source"),
                "published_date": article.get("published_date"),
                "url": article.get("url"),
                "text": chunk
            })

    print("Encoding embeddings...")
    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity
    index.add(embeddings)

    print("Saving index...")
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Index built successfully.")


# ==========================================
# LOAD INDEX
# ==========================================

def load_index():
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading metadata...")
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model


# ==========================================
# SEARCH
# ==========================================

def search(query, index, metadata, model, top_k=5):

    # E5 nécessite ce format
    formatted_query = "query: " + query

    query_embedding = model.encode(
        [formatted_query],
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        result = metadata[idx].copy()
        result["score"] = float(score)
        results.append(result)

    return results


# ==========================================
# MAIN
# ==========================================

def main():

    if not os.path.exists(INDEX_PATH):
        print("Index not found. Building index...")
        data = load_dataset(DATA_PATH)
        build_index(data)

    index, metadata, model = load_index()

    print("\nSystem ready. Type a query (or 'exit'):\n")

    while True:
        query = input(">> ")

        if query.lower() == "exit":
            break

        results = search(query, index, metadata, model, TOP_K)

        print("\nTop results:\n")
        for r in results:
            print("Score:", round(r["score"], 4))
            print("Title:", r["title"])
            print("Source:", r["source"])
            print("Date:", r["published_date"])
            print("URL:", r["url"])
            print("-" * 60)


if __name__ == "__main__":
    main()
