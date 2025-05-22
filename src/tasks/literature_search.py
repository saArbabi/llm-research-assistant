import json

import requests
from rich.console import Console

from tasks import MOCK_BEHAVIOR

console = Console()


def search_semantic_scholar(query, limit=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,abstract,url"}

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    papers = []
    for paper in data.get("data", []):
        title = paper.get("title", "No Title")
        abstract = paper.get("abstract", "No Abstract Available")
        url = paper.get("url", "")
        papers.append({"title": title, "abstract": abstract, "url": url})
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
        return mock_literature_search()

    search_results = []
    for query in queries:
        try:
            console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}")
            search_results.extend(search_semantic_scholar(query))
        except Exception as ex:
            console.print(f"[bold red]Search error for query '{query}':[/bold red] {str(ex)}")

    return remove_missing_abstracts(search_results)


def mock_literature_search():
    return remove_missing_abstracts(load_jsonl("data/testing_search.jsonl"))


def save_papers_to_jsonl(papers, filename="papers.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")
    print(f"Saved {len(papers)} papers to '{filename}'.")


def load_jsonl(filename="papers.jsonl"):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data
