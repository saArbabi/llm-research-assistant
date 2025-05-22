from openai_tools import create_openai_client


async def generate_report(
    user_description: str,
    proposed_solution: str,
    relevant_literature: list[dict],
    model: str = "gpt-3.5-turbo",
) -> str:

    system_prompt = (
        system_prompt
    ) = """
        You are a research report writer. You will be given a problem statement, a proposed solution, a paper title, an abstract, and links to the most relevant literature, along with a structured list of relevant literature entries.

        Your task is to create a comprehensive research report containing the following sections:

        1. **Proposed Solution** – Summarize and present the proposed solution clearly and concisely.
        2. **Novelty Assessment** – Critically evaluate the novelty of the proposed solution in comparison to the cited literature. Highlight similarities, differences, and any incremental or groundbreaking contributions.
        3. **Suggested Experiments** – Propose a set of practical experiments or studies to validate the proposed solution, detailing the methodology and expected outcomes.

        **Formatting Requirements:**
        - The report should be **well-structured**, **informative**, and **concise** (aim for 1-2 pages).
        - Use **Markdown** formatting.
        - Include a **Table of Contents** with internal links to each section.
        - Use **headings** and **subheadings** for readability.
        - Include **in-text citations** (e.g., “[Author, Year]”) with a **Source List** at the end.
        - Focus on **actionable insights** and **practical information** rather than academic verbosity.
        """

    user_prompt = f"""
            Problem Statement:
            {user_description}

            Proposed Solution:
            {proposed_solution}

            Relevant Literature:
            {format_similarity_results(relevant_literature)}

            Please write a research report according to the instructions in the system prompt.
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
    return search_query


def format_similarity_results(relevant_literature) -> str:
    formatted = []
    for i, result in enumerate(relevant_literature, 1):
        formatted.append(
            f"Title: {result.title}\n" f"Abstract: {result.abstract}\n" f"URL: {result.url}\n"
        )
    return "\n".join(formatted)
