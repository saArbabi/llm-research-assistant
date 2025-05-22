from pydantic import BaseModel

from openai_tools import create_openai_client, parse_llm_response
from tasks import MOCK_BEHAVIOR


async def generate_search_ideas(user_description: str, model: str = "gpt-3.5-turbo") -> str:

    if MOCK_BEHAVIOR["generate_search_ideas"]:
        return mock_generate_search_ideas()

    system_prompt = """
    You are a helpful assistant that generates ideas for research based on a user description.
    The description you will be given is a technical problem statement.

    You MUST return your response in a **strictly valid JSON format** using this structure:

    {
    "idea": "description of your proposed solution",
    "abstract": "a research abstract for a potential article that would address the user's described problem"
    }

    Ensure:
    - All keys and string values use double quotes ("")
    - No trailing commas
    - Do not include markdown formatting
    - The response must be a parsable JSON object only

    Think carefully and internally reason through how you would solve the problem based on your knowledge.
    Do NOT include your reasoning in the final output—only return the JSON object.

    Example description:
    Current image generation models often struggle with accurately generating hands. How could I go about improving these models to fix this?

    Example response:
    {
    "idea": "One solution to improve the accuracy of generated hands in image generation models is to incorporate explicit hand pose supervision into the training process using a pre-trained hand pose estimator. This strategy combines generative diffusion models with auxiliary guidance from a network that specializes in understanding human hand structure. During training, the model is not only penalized for poor visual fidelity but also for generating hand poses that deviate from realistic skeletal configurations, as predicted by the hand pose estimator.",
    "abstract": "Generative models, particularly diffusion-based architectures, have achieved impressive results in photorealistic image synthesis. However, they continue to struggle with generating anatomically accurate human hands, frequently producing distorted or implausible configurations. This paper presents a novel training augmentation strategy that leverages pre-trained hand pose estimators to enforce anatomical correctness during image generation. We introduce a dual-loss mechanism wherein, alongside conventional image reconstruction or denoising objectives, a hand pose loss penalizes deviations from realistic hand skeletal structures as predicted from generated images. Our method seamlessly integrates into existing diffusion pipelines and requires no changes at inference time. We evaluate our approach on hand-focused image synthesis benchmarks and demonstrate significant improvements in hand realism, measured via both perceptual studies and quantitative metrics such as Fréchet Inception Distance (FID) on hand-cropped regions and 3D pose consistency scores. Our results indicate that pose-aware supervision is a promising direction for mitigating one of the key failure cases in modern image generation models."
    }
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
    return parse_llm_response(search_query, LLMSolutionResponse)


def mock_generate_search_ideas():
    # Sample LLM response
    dummy_response_text = """
        {
        "idea": "To reduce hallucinations in language model-generated summaries of scientific literature, one approach is to incorporate citation-aware fine-tuning using a dataset of abstracts and full-text documents with aligned citations. During training, the model is encouraged to generate content that is explicitly supported by cited sources. A secondary verification model can also be introduced post-generation to flag and filter unsupported claims by cross-referencing the generated summary with the original paper.",
        "abstract": "Large language models have shown promise in summarizing scientific literature, but often generate factual inaccuracies or hallucinations that undermine trust. In this work, we propose a citation-aware training framework that grounds summary generation in verifiable content from the source text. Our approach involves fine-tuning a transformer-based model on a curated dataset of scientific papers where each sentence in the abstract is linked to supporting content in the body of the paper. Additionally, we introduce a verification module that evaluates factual alignment between the generated summary and the original document using semantic similarity and citation overlap. Experimental results on benchmark scientific summarization datasets demonstrate a significant reduction in unsupported claims, improving factual precision without sacrificing fluency. This research highlights the importance of grounding mechanisms in developing trustworthy scientific AI systems."
        }
    """
    return parse_llm_response(dummy_response_text, LLMSolutionResponse)


class LLMSolutionResponse(BaseModel):
    idea: str
    abstract: str
