from src.openai_tools import create_openai_client


def generate_search_query(user_description: str, model: str = "gpt-3.5-turbo") -> str:

    system_prompt = (
        "You are an academic research assistant. "
        "Your task is to convert a user's research question or problem into a concise academic search query. "
        "Return only the search query, with no explanation, punctuation, or complete sentences. "
        "Use only technical terms and keywords likely to appear in academic paper titles or abstracts. "
        "Avoid verbs, stopwords, or conversational phrases such as 'how to' or 'ways to'. "
        "Focus on nouns, domain-specific terms, and established concepts."
    )

    client = create_openai_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_description},
        ],
        max_tokens=30,
    )
    search_query = response.choices[0].message.content.strip()
    return search_query


if __name__ == "__main__":
    user_description = (
        "I'm trying to find methods for reducing bias in large language models during fine-tuning."
    )
    # e.g., output: methods reducing bias large language models fine-tuning
    search_query = generate_search_query(user_description)
    print(search_query)
