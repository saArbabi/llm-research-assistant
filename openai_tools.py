import os

import numpy as np
import openai
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

logger.remove()
logger.add(
    sink=lambda message: print(message), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def check_openai_api_key(client):
    try:
        client.models.list()
    except openai.AuthenticationError:
        logger.error("Invalid API key")
    else:
        logger.success("API key validated successfully")


def get_embedding(abstracts, model="text-embedding-3-small"):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    check_openai_api_key(client)

    embeddings = {}
    for abstract in abstracts:
        if len(abstract) > 20: 
            logger.info(f"Generating embeddings for '{abstract[:20]} ...'")
        else:
            logger.info(f"Generating embeddings for {abstract}")
        response = client.embeddings.create(input=abstract, model=model)
        embeddings[abstract] = np.array(response.data[0].embedding)
    logger.success("Embeddings generated")
    return embeddings
