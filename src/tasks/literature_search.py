import os
import time

import requests
from loguru import logger
from requests.exceptions import RequestException
from rich.console import Console

from tasks import MOCK_BEHAVIOR
from tasks.utils import load_jsonl, write_jsonl

console = Console()
logger.remove()
logger.add(
    sink=lambda message: print(message), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def search_semantic_scholar(queries: list[str], limit: int = 10) -> list[dict[str, str]]:

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    papers = []
    for query in queries:
        try:
            console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}")
            params = {"query": query, "limit": limit, "fields": "title,abstract,url"}

            response = requests.get(url, params=params, timeout=10)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))  # default wait time: 10 sec
                console.print(f"[yellow]Rate limit exceeded. Retrying after {retry_after} seconds...[/yellow]")
                time.sleep(retry_after)
                response = requests.get(url, params=params, timeout=10)

            response.raise_for_status()

            data = response.json()
            for paper in data.get("data", []):
                title = paper.get("title", "No Title")
                abstract = paper.get("abstract", "No Abstract Available")
                paper_url = paper.get("url", "")
                papers.append({"title": title, "abstract": abstract, "url": paper_url})

        except RequestException as ex:
            status_code = getattr(ex.response, "status_code", "N/A")
            console.print(
                f"[bold red]Request error for query '{query}': {ex.__class__.__name__} - {str(ex)} "
                f"(HTTP {status_code})[/bold red]"
            )

    return papers


def remove_missing_abstracts(raw_search_results):
    search_results = []
    for res in raw_search_results:
        if not res["abstract"]:
            continue
        search_results.append(res)
    return search_results


def perform_literature_search(queries):
    if MOCK_BEHAVIOR["perform_literature_search"]:
        return mock_literature_search(queries)

    search_results = search_semantic_scholar(queries)
    return remove_missing_abstracts(search_results)


def mock_literature_search(queries):
    file_path = os.path.join("mock_data", "example_papers.jsonl")
    if not os.path.exists(file_path):
        papers = search_semantic_scholar(queries)
        write_jsonl(papers, file_path)
        logger.info(f"Saved {len(papers)} papers to '{file_path}'.")
    else:
        papers = load_jsonl(file_path)
    return remove_missing_abstracts(papers)