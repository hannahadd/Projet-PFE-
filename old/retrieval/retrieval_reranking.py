import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "merged_dataset.json"
INDEX_PATH = "faiss_index.bin"
META_PATH = "metadata.json"

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 350
TOP_K_RETRIEVAL = 20   # candidats initiaux FAISS
TOP_K_FINAL = 5        # résultats après reranking


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

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    documents = []
    metadata = []

    print("Chunking documents...")
    for article in tqdm(data):

        content = article.get("content", "")
        if not content:
            continue

        chunks = chunk_text(content, CHUNK_SIZE)

        for chunk in chunks:
            documents.append("passage: " + chunk)  # format E5
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
    index = faiss.IndexFlatIP(dimension)  # cos sim via normalized vectors
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

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Loading reranker model...")
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    return index, metadata, embedding_model, reranker


# ==========================================
# SEARCH WITH RERANKING
# ==========================================

def search(query, index, metadata,
           embedding_model, reranker,
           top_k_retrieval=20,
           top_k_final=5):

    # Format obligatoire pour E5
    formatted_query = "query: " + query

    query_embedding = embedding_model.encode(
        [formatted_query],
        normalize_embeddings=True
    )

    # Étape 1 — Retrieval large
    scores, indices = index.search(query_embedding, top_k_retrieval)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        candidate = metadata[idx].copy()
        candidate["faiss_score"] = float(score)
        candidates.append(candidate)

    # Étape 2 — Reranking cross-encoder
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)

    # Tri final par rerank_score
    candidates = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return candidates[:top_k_final]


# ==========================================
# MAIN
# ==========================================

def main():

    if not os.path.exists(INDEX_PATH):
        print("Index not found. Building index...")
        data = load_dataset(DATA_PATH)
        build_index(data)

    index, metadata, embedding_model, reranker = load_index()

    print("\nSystem ready. Type a query (or 'exit'):\n")

    while True:

        query = input(">> ")

        if query.lower() == "exit":
            break

        results = search(
            query,
            index,
            metadata,
            embedding_model,
            reranker,
            TOP_K_RETRIEVAL,
            TOP_K_FINAL
        )

        print("\nTop results after reranking:\n")

        for r in results:
            print("Rerank score:", round(r["rerank_score"], 4))
            print("FAISS score:", round(r["faiss_score"], 4))
            print("Title:", r["title"])
            print("Source:", r["source"])
            print("Date:", r["published_date"])
            print("URL:", r["url"])
            print("-" * 60)


if __name__ == "__main__":
    main()