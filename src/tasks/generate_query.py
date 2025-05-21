from src.openai_tools import create_openai_client


def generate_search_query(user_description: str, model: str = "gpt-3.5-turbo") -> str:

    system_prompt = (
        "You are an advanced academic research strategist. "
        "Your task is to help users turn a complex research question or problem into effective academic search queries. "
        "First, analyze the input by identifying the key concepts, subtopics, and domain-specific terminology involved. "
        "Break down what needs to be researched, consider potential challenges such as ambiguity or scope, "
        "and briefly explain your strategy for finding comprehensive and high-quality academic information. "
        "Then, generate 3 academic search queries that meet the following criteria: "
        "they are highly specific, use only technical terms or domain-relevant keywords, and reflect different aspects of the original research question. "
        "Avoid complete sentences, punctuation, stopwords, and general-purpose verbs. "
        "Each query should resemble a set of terms found in academic titles or abstracts."
        "Make sure to provide both your thought process and the generated queries"
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
