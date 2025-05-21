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
        logger.error("Invalid API key.")
    except openai.PermissionError:
        logger.error("Permission denied. Check your API key's access scope.")
    except openai.RateLimitError:
        logger.error("Rate limit exceeded. Please wait and try again.")
    except openai.APIConnectionError:
        logger.error("Network error: Could not connect to OpenAI servers.")
    except openai.InvalidRequestError as e:
        logger.error(f"Invalid request: {e}")
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    else:
        logger.info("API key validated successfully.")


def create_openai_client():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    check_openai_api_key(client)
    return client


def get_embedding(abstracts, model="text-embedding-3-small"):
    client = create_openai_client()

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
    try:
        if isinstance(response_text, str):
            parsed = json.loads(response_text)
        elif isinstance(response_text, dict):
            parsed = response_text
        else:
            raise ValueError("LLM output is not in expected format")
        return model.model_validate(parsed)

    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Failed to parse or validate LLM response: {e}")
