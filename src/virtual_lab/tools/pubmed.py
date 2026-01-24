"""PubMed search and article retrieval tools.

This module provides functions for searching PubMed Central and retrieving
article content for use in AI-assisted research discussions.

Usage:
    from virtual_lab.tools.pubmed import run_pubmed_search

    articles = run_pubmed_search("CRISPR gene editing", num_articles=3)
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

import requests
from openai.types.beta.threads.run import Run

from virtual_lab.config.constants import (
    PUBMED_BASE_URL,
    PUBMED_SEARCH_URL,
    PUBMED_TOOL_NAME,
    SECTION_TYPES,
    ABSTRACT_ONLY_SECTIONS,
)
from virtual_lab.logging_config import get_logger
from virtual_lab.prompts import format_references

logger = get_logger(__name__)


def get_pubmed_central_article(
    pmcid: str, abstract_only: bool = False
) -> tuple[str | None, list[str] | None]:
    """Get the title and content of a PubMed Central article by PMC ID.

    Retrieves the article text in a structured format, filtering for relevant
    sections (abstract, intro, results, discussion, conclusions, methods).

    Note: This only returns main text, ignoring tables, figures, and references.

    Args:
        pmcid: The PMC ID of the article (without the "PMC" prefix).
        abstract_only: If True, return only the abstract instead of full text.

    Returns:
        A tuple of (title, content) where content is a list of paragraph strings.
        Returns (None, None) if the article is not found or cannot be parsed.

    Example:
        title, paragraphs = get_pubmed_central_article("1234567")
        if title:
            print(f"Article: {title}")
            print(f"Paragraphs: {len(paragraphs)}")
    """
    text_url = f"{PUBMED_BASE_URL}/BioC_JSON/PMC{pmcid}/unicode"

    try:
        response = requests.get(text_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch article PMC{pmcid}: {e}")
        return None, None

    try:
        article = response.json()
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON for article PMC{pmcid}")
        return None, None

    try:
        document = article[0]["documents"][0]

        # Get title
        title = next(
            passage["text"]
            for passage in document["passages"]
            if passage["infons"]["section_type"] == "TITLE"
        )

        # Get relevant passages
        passages = [
            passage
            for passage in document["passages"]
            if passage["infons"]["type"] in {"abstract", "paragraph"}
        ]

        # Filter by section type
        section_filter = ABSTRACT_ONLY_SECTIONS if abstract_only else SECTION_TYPES
        passages = [
            passage
            for passage in passages
            if passage["infons"]["section_type"] in section_filter
        ]

        content = [passage["text"] for passage in passages]
        return title, content

    except (KeyError, IndexError, StopIteration) as e:
        logger.warning(f"Failed to extract content from article PMC{pmcid}: {e}")
        return None, None


def run_pubmed_search(
    query: str, num_articles: int = 3, abstract_only: bool = False
) -> str:
    """Search PubMed Central and return formatted article content.

    Performs a relevance-sorted search on PubMed Central and retrieves the
    full text (or abstracts) of the top matching articles.

    Args:
        query: The search query string.
        num_articles: Maximum number of articles to return.
        abstract_only: If True, return only abstracts instead of full text.

    Returns:
        A formatted string containing the article content, suitable for
        inclusion in an AI conversation.

    Example:
        results = run_pubmed_search("CRISPR applications", num_articles=2)
        print(results)
    """
    content_type = "abstracts" if abstract_only else "full text"
    logger.info(
        f'Searching PubMed Central for {num_articles} articles ({content_type}) '
        f'with query: "{query}"'
    )

    # Perform search
    search_url = (
        f"{PUBMED_SEARCH_URL}?db=pmc"
        f"&term={urllib.parse.quote_plus(query)}"
        f"&retmax={2 * num_articles}"
        f"&retmode=json"
        f"&sort=relevance"
    )

    try:
        response = requests.get(search_url, timeout=30)
        response.raise_for_status()
        pmcids_found = response.json()["esearchresult"]["idlist"]
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        logger.error(f"PubMed search failed: {e}")
        return f'Search failed for query "{query}": {e}'

    # Fetch articles
    texts: list[str] = []
    titles: list[str] = []
    pmcids: list[str] = []

    for pmcid in pmcids_found:
        if len(pmcids) >= num_articles:
            break

        title, content = get_pubmed_central_article(
            pmcid=pmcid,
            abstract_only=abstract_only,
        )

        if title is None:
            continue

        texts.append(f"PMCID = {pmcid}\n\nTitle = {title}\n\n{chr(10).join(content)}")
        titles.append(title)
        pmcids.append(pmcid)

    article_count = len(texts)
    logger.info(f"Found {article_count:,} articles on PubMed Central")

    if article_count == 0:
        return f'No articles found on PubMed Central for the query "{query}".'

    return format_references(
        references=tuple(texts),
        reference_type="paper",
        intro=f'Here are the top {article_count} articles on PubMed Central for the query "{query}":',
    )


def run_tools(run: Run) -> list[dict[str, str]]:
    """Execute tools required by a run and return their outputs.

    Processes the tool calls in a run's required_action and executes the
    appropriate tool functions.

    Args:
        run: The Run object with required_action containing tool calls.

    Returns:
        A list of tool output dictionaries with 'tool_call_id' and 'output' keys.

    Raises:
        ValueError: If an unknown tool is requested.
    """
    tool_outputs: list[dict[str, str]] = []

    for tool in run.required_action.submit_tool_outputs.tool_calls:
        if tool.function.name == PUBMED_TOOL_NAME:
            args_dict = json.loads(tool.function.arguments)
            output = run_pubmed_search(**args_dict)
            tool_outputs.append({
                "tool_call_id": tool.id,
                "output": output,
            })
        else:
            raise ValueError(f"Unknown tool: {tool.function.name}")

    return tool_outputs


def search_pubmed_for_context(
    topics: list[str],
    num_articles_per_topic: int = 2,
    abstract_only: bool = True,
) -> str:
    """Search PubMed for multiple topics and combine results.

    Convenience function for gathering context from multiple search queries,
    useful for building comprehensive background information.

    Args:
        topics: List of search queries/topics.
        num_articles_per_topic: Number of articles to fetch per topic.
        abstract_only: If True, fetch only abstracts for faster retrieval.

    Returns:
        Combined formatted string with all article content.

    Example:
        context = search_pubmed_for_context(
            ["CRISPR delivery methods", "AAV gene therapy"],
            num_articles_per_topic=3,
        )
    """
    all_results: list[str] = []

    for topic in topics:
        result = run_pubmed_search(
            query=topic,
            num_articles=num_articles_per_topic,
            abstract_only=abstract_only,
        )
        all_results.append(f"### Topic: {topic}\n\n{result}")

    return "\n\n---\n\n".join(all_results)
