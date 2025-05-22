import json
import pickle

import faiss
import numpy as np
from loguru import logger

from openai_tools import get_embedding
from tasks import MOCK_BEHAVIOR

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


def create_faiss_index(embeddings):
    logger.info("Creating FAISS index")
    index = faiss.IndexFlatL2(DIMENSION)
    index.add(np.array(list(embeddings.values())).astype("float32"))

    logger.success("FAISS index created")
    return index


def search(query, llm_search_results, top_k=3):
    if MOCK_BEHAVIOR["search"]:
        return mock_search(query, llm_search_results, top_k)


#     # get embedding for the literature abstract
#     abstracts = [res["abstract"] for res in llm_search_results]
#     abstract_embeddings = get_embedding(abstracts)
#     # get embedding for the llm_idea
#     query_embedding = get_embedding([query])
#     # create faiss index
#     faiss_index = create_faiss_index(abstract_embeddings)

# distances, indices = faiss_index.search(query_embedding, top_k)

# for i, idx in enumerate(indices[0]):
#     print(f"Match {i+1}:")
#     print(f"Abstract: {abstract_embeddings[idx][:50]} ...")
#     print(f"Distance: {distances[0][i]:.4f}\n")


def mock_search(query, llm_search_results, top_k=3):
    import os

    example_embeddings_path = os.path.join("data", "example_embeddings.pkl")
    example_query_embedding_path = os.path.join("data", "example_query_embedding.pkl")
    faiss_index_path = os.path.join("data", "faiss_index.index")
    abstracts = [res["abstract"] for res in llm_search_results]
    if not os.path.exists(example_embeddings_path):
        abstract_embeddings = get_embedding(abstracts)
        with open(example_embeddings_path, "wb") as f:
            pickle.dump(abstract_embeddings, f)
    else:
        with open(example_embeddings_path, "rb") as f:
            abstract_embeddings = pickle.load(f)

    if not os.path.exists(faiss_index_path):
        faiss_index = create_faiss_index(abstract_embeddings)
        faiss.write_index(faiss_index, faiss_index_path)
    else:
        faiss_index = faiss.read_index(faiss_index_path)

    if not os.path.exists(example_query_embedding_path):
        query_embedding = get_embedding([query])
        with open(example_query_embedding_path, "wb") as f:
            pickle.dump(query_embedding, f)
    else:
        with open(example_query_embedding_path, "rb") as f:
            query_embedding = pickle.load(f)
    distances, indices = faiss_index.search(query_embedding[query].reshape(1, -1), top_k)

    for i, idx in enumerate(indices[0]):
        print(f"Match {i+1}:")
        print(f"Abstract: {abstracts[idx][:50]} ...")
        print(f"Distance: {distances[0][i]:.4f}\n")
