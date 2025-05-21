from pydantic import BaseModel, field_validator

from openai_tools import create_openai_client, parse_llm_response


async def generate_search_query(user_description: str, model: str = "gpt-3.5-turbo") -> str:

    system_prompt = """
        You are a helpful assistant that generates search queries for research based on a user description. 

        You MUST return your response in a **valid JSON format** using this structure:

        {
        "queries": ["query1", "query2", "query3"],
        "thoughts": "Your explanation here"
        }

        First, reason through your strategy for retrieving high-quality information from literature. 
        Then, generate 3 specific and focused search queries. Only return the JSON object — do not include explanations outside of it.
        """
    user_prompt = f"""
        User description: {user_description}

        Follow the instructions in the system prompt and respond in the required JSON format.
        """

    client = create_openai_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    search_query = response.choices[0].message.content.strip()
    return parse_llm_response(search_query, QueryResponse)


async def mock_search_query():
    # Sample LLM response
    dummy_response_text = """
        {
            "queries": ["latest trends in robotics research", "impact of robotics on industrial automation", "future applications of robotics technology"],
            "thoughts": "These queries will provide insights into the current advancements, practical implications, and potential future developments in the field of robotics."
        }
    """
    return parse_llm_response(dummy_response_text, QueryResponse)


class QueryResponse(BaseModel):
    queries: list[str]
    thoughts: str
