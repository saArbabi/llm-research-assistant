import json
import os
import pickle

import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

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


def check_openai_api_key(client):
    try:
        client.models.list()
    except openai.AuthenticationError:
        logger.error("Invalid API key")
    else:
        logger.success("API key validated successfully")


def get_embedding(client, text, model="text-embedding-3-small"):
    logger.info("Generating embeddings")
    response = client.embeddings.create(input=text, model=model)
    logger.success("Embeddings generated")
    return np.array(response.data[0].embedding)


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
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    check_openai_api_key(client)
    embeddings = {text: get_embedding(client, text) for text in abstracts}
    build_and_save_index(embeddings)
