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
            "idea": "One approach to enhance the accuracy of hand generation in image models is to employ a multi-stage refinement process focused specifically on hand details. This process involves a cascaded architecture where initial image generation is followed by dedicated hand refinement modules that iteratively enhance the hand region at different levels of detail. These modules can incorporate hand-specific constraints, such as geometric priors, to ensure more realistic and anatomically correct hand synthesis.",
            "abstract": "Generating photorealistic images of hands remains challenging for existing image synthesis models due to the complexity of hand anatomy and fine details. This paper introduces a novel multi-stage refinement strategy tailored to improving hand generation in image models. Our approach integrates specialized hand refinement modules within the image generation pipeline to iteratively enhance the realism and accuracy of synthesized hands. These modules leverage hand-specific constraints, such as anatomical priors and geometric considerations, to guide the generation process towards more realistic results. We evaluate our method on benchmark datasets focusing on hand synthesis tasks and demonstrate substantial improvements in the quality of generated hand images, as evaluated by both subjective human assessment and objective quality metrics. Our results showcase the potential of incorporating dedicated hand refinement stages to address the challenges associated with generating lifelike hands in image synthesis models."
            }

    """
    # dummy_response_text = """
    #         {
    #         "idea": "One approach to enhance the accuracy of hand generation in image models is to employ a multi-stage refinement process focused specifically on hand details. This process involves a cascaded architecture where initial image generation is followed by dedicated hand refinement modules that iteratively enhance the hand region at different levels of detail. These modules can incorporate hand-specific constraints, such as geometric priors, to ensure more realistic and anatomically correct hand synthesis.",
    #         "abstract": "Text-to-image generation models have achieved remarkable advancements in recent years, aiming to produce realistic images from textual descriptions. However, these models often struggle with generating anatomically accurate representations of human hands. The resulting images frequently exhibit issues such as incorrect numbers of fingers, unnatural twisting or interlacing of fingers, or blurred and indistinct hands. These issues stem from the inherent complexity of hand structures and the difficulty in aligning textual descriptions with precise visual depictions of hands. To address these challenges, we propose a novel approach named Hand1000 that enables the generation of realistic hand images with target gesture using only 1,000 training samples. The training of Hand1000 is divided into three stages with the first stage aiming to enhance the model’s understanding of hand anatomy by using a pre-trained hand gesture recognition model to extract gesture representation. The second stage further optimizes text embedding by incorporating the extracted hand gesture representation, to improve alignment between the textual descriptions and the generated hand images. The third stage utilizes the optimized embedding to fine-tune the Stable Diffusion model to generate realistic hand images. In addition, we construct the first publicly available dataset specifically designed for text-to-hand image generation. Based on the existing hand gesture recognition dataset, we adopt advanced image captioning models and LLaMA3 to generate high-quality textual descriptions enriched with detailed gesture information. Extensive experiments demonstrate that Hand1000 significantly outperforms existing models in producing anatomically correct hand images while faithfully representing other details in the text, such as faces, clothing and colors."
    #         }

    # """
    return parse_llm_response(dummy_response_text, LLMSolutionResponse)


class LLMSolutionResponse(BaseModel):
    idea: str
    abstract: str
