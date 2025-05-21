import json

import requests


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


def save_papers_to_jsonl(papers, filename="papers.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")
    print(f"Saved {len(papers)} papers to '{filename}'.")
