## Overview of the approach

This package uses a two-branch setup to support research ideation and novelty evaluation by combining large language models (LLMs) with retrieval tools.

1. **Generative branch:**  
   This branch starts by generating research ideas based only on the user’s description of a problem. It doesn't reference existing literature to encourage more original solutions. These ideas are then passed to the next branch for evaluation.

2. **Retrieval + evaluation branch:**  
   This branch searches existing research (currently using abstracts) to check how novel the generated ideas are. It uses semantic similarity (via FAISS) to compare new ideas against academic literature, suggesting experiments for solution validation.



### Why this structure

LLMs are good at generating new ideas because they’ve been trained on a wide range of human knowledge. They can combine concepts.
By having the LLM generate ideas without being influenced by existing papers, the aim to take advantage of that creative potential. Then, by checking those ideas against real research, we keep things grounded and relevant.

## Features
- **Query Generation**: Automatically formulates relevant search queries from your problem statement.
- **Literature Retrieval**: Searches and parses abstracts from academic literature related to the topic.
- **Idea Generation**: Proposes new research ideas.
- **Novelty Assessment**: Compares generated ideas against retrieved literature to assess originality using similarity search.
- **Report Generation**: Produces a structured summary report that includes:
  - Proposed research ideas
  - Novelty 
  - Suggested experiments for validation

![alt text](solution_diagram.jpg "Title")
## Setup instructions



### 1. Create a virtual environment

Run this in the project directory:
```bash
python -m venv <venv>
```

### 3. Activate the virtual environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install dependencies

Install the reuired packages in `requirements.txt`:
```bash
pip install -r requirements.txt
```
### 5. Set API key as an environment variable
Create a `.env` file in the root directory with your API key secrets:
```
OPENAI_API_KEY=your-api-key-here
```
See [Open AI](https://platform.openai.com/api-keys) for how to create keys.

## Usage

Run the assistant:
```bash
python src/main.py
```

You'll be prompted to enter a technical problem description. Example:
```
Current image generation models often struggle with accurately generating hands. How could I go about improving these models to fix this?
```

The assistant will respond with relevant research directions, novelty checks, and experiment suggestions.

## Future improvements
1. **Improve literature retrieval**  
   Currently, the package retrieves information from paper abstracts. A more advanced approach would involve implementing a RAG to extract richer, context-aware information directly from full-text literature.

2. **LLM function calling and agentic behavior**  
   Introduce more dynamic agent capabilities using LLM function calling, such as:
   - Autonomous re-searching until a specified compute or token budget is met.
   - Retrospective analysis where the agent reviews its own trace, identifies gaps in retrieved knowledge, and generates additional, targeted queries to address those gaps.
