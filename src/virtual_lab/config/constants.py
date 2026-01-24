"""Constants for Virtual Lab.

This module centralizes all constant values used throughout the codebase,
including model pricing, API URLs, and default configuration values.
"""

from __future__ import annotations

# Default model for agents
DEFAULT_MODEL = "gpt-5-mini"

# API URLs
PUBMED_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Section types for PubMed article parsing
SECTION_TYPES = ["ABSTRACT", "INTRO", "RESULTS", "DISCUSS", "CONCL", "METHODS"]
ABSTRACT_ONLY_SECTIONS = ["ABSTRACT"]

# Temperature settings
CONSISTENT_TEMPERATURE = 0.2
CREATIVE_TEMPERATURE = 0.8

# Retry settings
MAX_RETRIES = 6
RETRY_BASE_DELAY = 0.6

# Fine-tuning settings
DEFAULT_FINETUNING_EPOCHS = 4

# Model pricing (USD per token) as of January 2025
# Source: https://openai.com/api/pricing/
MODEL_TO_INPUT_PRICE_PER_TOKEN: dict[str, float] = {
    "gpt-3.5-turbo-0125": 0.5 / 10**6,
    "gpt-4o-2024-08-06": 2.5 / 10**6,
    "gpt-4o-2024-05-13": 5 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.15 / 10**6,
    "o1-mini-2024-09-12": 3 / 10**6,
    "gpt-5-mini": 0.15 / 10**6,  # Estimated
    "gpt-5": 2.5 / 10**6,  # Estimated
}

MODEL_TO_OUTPUT_PRICE_PER_TOKEN: dict[str, float] = {
    "gpt-3.5-turbo-0125": 1.5 / 10**6,
    "gpt-4o-2024-08-06": 10 / 10**6,
    "gpt-4o-2024-05-13": 15 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.6 / 10**6,
    "o1-mini-2024-09-12": 12 / 10**6,
    "gpt-5-mini": 0.6 / 10**6,  # Estimated
    "gpt-5": 10 / 10**6,  # Estimated
}

FINETUNING_MODEL_TO_INPUT_PRICE_PER_TOKEN: dict[str, float] = {
    "gpt-4o-2024-08-06": 3.75 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.3 / 10**6,
}

FINETUNING_MODEL_TO_OUTPUT_PRICE_PER_TOKEN: dict[str, float] = {
    "gpt-4o-2024-08-06": 15 / 10**6,
    "gpt-4o-mini-2024-07-18": 1.2 / 10**6,
}

FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN: dict[str, float] = {
    "gpt-4o-2024-08-06": 25 / 10**6,
    "gpt-4o-mini-2024-07-18": 3 / 10**6,
}

# PubMed tool configuration
PUBMED_TOOL_NAME = "pubmed_search"
PUBMED_TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": PUBMED_TOOL_NAME,
        "description": "Get abstracts or the full text of biomedical and life sciences articles from PubMed Central.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to use to search PubMed Central for scientific articles.",
                },
                "num_articles": {
                    "type": "integer",
                    "description": "The number of articles to return from the search query.",
                },
                "abstract_only": {
                    "type": "boolean",
                    "description": "Whether to return only the abstract of the articles.",
                },
            },
            "required": ["query", "num_articles"],
        },
    },
}

# NIH Score anchors
NIH_SCORE_ANCHORS = {
    1: "Exceptional: Proposal demonstrates extraordinarily compelling importance and rigor with multiple major strengths that set it apart; essentially no weaknesses.",
    2: "Outstanding: Proposal has extremely strong strengths that decisively outweigh negligible weaknesses of the proposal; highly compelling overall.",
    3: "Excellent: Proposal is very strong, with several clear strengths; only a few minor weaknesses that do not diminish overall impact.",
    4: "Very Good: Proposal is strong and contains notable strengths, but numerous minor weaknesses reduce overall enthusiasm.",
    5: "Good: Proposal has identifiable strengths, but at least one moderate weakness lowers confidence in overall impact.",
    6: "Satisfactory: Proposal shows some strengths, but multiple moderate weaknesses substantially temper confidence in success.",
    7: "Fair: Proposal includes limited strengths, but at least one major weakness dominates the assessment.",
    8: "Marginal: Proposal has very few strengths and several major weaknesses that seriously compromise its impact.",
    9: "Poor: Proposal has virtually no strengths; numerous major weaknesses make it very unlikely to exert a positive impact.",
}

# Grant scoring criteria
GRANT_SCORING_CRITERIA = (
    "Significance",
    "Innovation",
    "Rigor",
    "Reproducibility",
    "Overall Impact",
)

# Workflow phases
DEFAULT_WORKFLOW_PHASES = [
    "team_selection",
    "independent_review",
    "collaboration_review",
    "chair_merge",
    "final_output",
]
