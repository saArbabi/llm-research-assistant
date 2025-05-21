import pickle

import faiss

from src.openai_tools import get_embedding

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536


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
