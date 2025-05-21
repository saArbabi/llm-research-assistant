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


if __name__ == "__main__":
    EMBEDDING_MODEL = "text-embedding-3-small"
    DIMENSION = 1536

    abstracts = load_abstracts(filename="papers.jsonl")

    embeddings = get_embedding(abstracts)
    build_and_save_index(embeddings)
