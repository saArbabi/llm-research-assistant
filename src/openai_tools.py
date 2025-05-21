import json
import os

import numpy as np
import openai
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, ValidationError

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


def create_openai_client():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


def get_embedding(abstracts, model="text-embedding-3-small"):
    client = create_openai_client()
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


def parse_llm_response(response_text: str, model: BaseModel) -> BaseModel:
    """
    Extracts JSON from an LLM response and validates it against a Pydantic model.

    Args:
        response_text (str): Raw response from the LLM, expected to contain JSON.
        model (BaseModel): A Pydantic model class to validate against.

    Returns:
        An instance of the Pydantic model with validated data.

    Raises:
        ValueError: If JSON extraction or validation fails.
    """
    try:
        # Attempt to find the first JSON object in the response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]

        data = json.loads(json_str)
        return model.model_validate(data)

    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Failed to parse or validate LLM response: {e}")
