import json
import pickle

import faiss
import numpy as np
from loguru import logger

from src.openai_tools import get_embedding

logger.remove()
logger.add(
    sink=lambda message: print(message), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536


def load_abstracts(filename="papers.jsonl"):
    abstracts = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            abstracts.append(data["abstract"])
    return abstracts


def build_and_save_index(embeddings, index_path="faiss.index", metadata_path="metadata.pkl"):
    logger.info("Creating FAISS index")
    index = faiss.IndexFlatL2(DIMENSION)
    index.add(np.array(list(embeddings.values())).astype("float32"))

    logger.success("FAISS index created")
    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.success("FAISS index and metadata saved successfully")


def get_query_embedding(query):
    embedding = get_embedding([query])
    return embedding[query]


def search(query, index_path="faiss.index", metadata_path="metadata.pkl", top_k=3):
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        embedding_cache = list(pickle.load(f).keys())

    query_vec = get_query_embedding(query).reshape(1, -1)

    distances, indices = index.search(query_vec, top_k)

    for i, idx in enumerate(indices[0]):
        print(f"Match {i+1}:")
        print(f"Abstract: {embedding_cache[idx][:50]} ...")
        print(f"Distance: {distances[0][i]:.4f}\n")


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    search(user_query)
