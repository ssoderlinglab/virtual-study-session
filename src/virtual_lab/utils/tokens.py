"""Token counting and cost calculation utilities.

This module provides functions for counting tokens in text and calculating
API costs based on token usage.

Usage:
    from virtual_lab.utils.tokens import count_tokens, compute_token_cost

    tokens = count_tokens("Hello, world!")
    cost = compute_token_cost("gpt-5-mini", input_tokens=1000, output_tokens=500)
"""

from __future__ import annotations

import tiktoken

from virtual_lab.config.constants import (
    MODEL_TO_INPUT_PRICE_PER_TOKEN,
    MODEL_TO_OUTPUT_PRICE_PER_TOKEN,
    FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN,
    DEFAULT_FINETUNING_EPOCHS,
)
from virtual_lab.logging_config import get_logger

logger = get_logger(__name__)


def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """Return the number of tokens in a text string.

    Args:
        string: The text string to count tokens in.
        encoding_name: The name of the tiktoken encoding to use.

    Returns:
        The number of tokens in the text string.

    Example:
        >>> count_tokens("Hello, world!")
        4
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def update_token_counts(
    token_counts: dict[str, int],
    discussion: list[dict[str, str]],
    response: str,
) -> None:
    """Update token counts in place with a discussion and response.

    This function calculates input tokens from the discussion history and
    output tokens from the response, then updates the provided dictionary.

    Args:
        token_counts: The token counts dictionary to update (modified in place).
            Expected keys: 'input', 'output', 'max'.
        discussion: The discussion history to count as input tokens.
        response: The response to count as output tokens.
    """
    new_input_token_count = sum(count_tokens(turn["message"]) for turn in discussion)
    new_output_token_count = count_tokens(response)

    token_counts["input"] += new_input_token_count
    token_counts["output"] += new_output_token_count

    token_counts["max"] = max(
        token_counts["max"], new_input_token_count + new_output_token_count
    )


def count_discussion_tokens(
    discussion: list[dict[str, str]],
) -> dict[str, int]:
    """Count the number of tokens in a discussion.

    Iterates through the discussion and calculates cumulative input/output
    token counts as if each message were generated sequentially.

    Args:
        discussion: The discussion to count tokens in.
            Each entry should have 'agent' and 'message' keys.

    Returns:
        A dictionary with keys 'input', 'output', and 'max'.
    """
    token_counts = {
        "input": 0,
        "output": 0,
        "max": 0,
    }

    for index, turn in enumerate(discussion):
        if turn["agent"] != "User":
            update_token_counts(
                token_counts=token_counts,
                discussion=discussion[:index],
                response=turn["message"],
            )

    return token_counts


def compute_token_cost(
    model: str, input_token_count: int, output_token_count: int
) -> float:
    """Compute the cost in USD for token usage with a given model.

    Args:
        model: The name of the model.
        input_token_count: The number of input tokens.
        output_token_count: The number of output tokens.

    Returns:
        The cost in USD.

    Raises:
        ValueError: If the model's pricing is not known.
    """
    if (
        model not in MODEL_TO_INPUT_PRICE_PER_TOKEN
        or model not in MODEL_TO_OUTPUT_PRICE_PER_TOKEN
    ):
        raise ValueError(f'Cost of model "{model}" not known')

    return (
        input_token_count * MODEL_TO_INPUT_PRICE_PER_TOKEN[model]
        + output_token_count * MODEL_TO_OUTPUT_PRICE_PER_TOKEN[model]
    )


def print_cost_and_time(
    token_counts: dict[str, int],
    model: str,
    elapsed_time: float,
) -> None:
    """Print token counts, cost, and elapsed time.

    Args:
        token_counts: Dictionary with 'input', 'output', 'tool', and 'max' keys.
        model: The model name for cost calculation.
        elapsed_time: Elapsed time in seconds.
    """
    print(f"Input token count: {token_counts['input']:,}")
    print(f"Output token count: {token_counts['output']:,}")
    print(f"Tool token count: {token_counts.get('tool', 0):,}")
    print(f"Max token length: {token_counts['max']:,}")

    try:
        cost = compute_token_cost(
            model=model,
            input_token_count=token_counts["input"] + token_counts.get("tool", 0),
            output_token_count=token_counts["output"],
        )
        print(f"Cost: ${cost:.2f}")
    except ValueError as e:
        logger.warning(f"Could not compute cost: {e}")

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Time: {minutes}:{seconds:02d}")


def compute_finetuning_cost(
    model: str, token_count: int, num_epochs: int = DEFAULT_FINETUNING_EPOCHS
) -> float:
    """Compute the cost of fine-tuning a model.

    Args:
        model: The model that will be fine-tuned.
        token_count: The number of training tokens.
        num_epochs: Number of fine-tuning epochs.

    Returns:
        The cost of fine-tuning in USD.

    Raises:
        ValueError: If the model's fine-tuning pricing is not known.
    """
    if model not in FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN:
        raise ValueError(f'Fine-tuning cost of model "{model}" not known')

    return (
        token_count * FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN[model] * num_epochs
    )
