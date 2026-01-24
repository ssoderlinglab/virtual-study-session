"""Utility functions for Virtual Lab.

This package provides utilities organized by functionality:
- tokens: Token counting and cost calculation
- messages: Message conversion and retrieval
- io: File I/O operations

For backward compatibility, all functions are re-exported at the package level.
"""

from virtual_lab.utils.tokens import (
    count_tokens,
    update_token_counts,
    count_discussion_tokens,
    compute_token_cost,
    print_cost_and_time,
    compute_finetuning_cost,
)
from virtual_lab.utils.messages import (
    get_conversation_messages,
    get_messages,
    async_get_messages,
    convert_messages_to_discussion,
)
from virtual_lab.utils.io import (
    save_meeting,
    get_summary,
    load_summaries,
    get_recent_markdown,
    write_final_summary,
    clear_dir,
    check_files,
    load_json,
    save_json,
)

__all__ = [
    # Token utilities
    "count_tokens",
    "update_token_counts",
    "count_discussion_tokens",
    "compute_token_cost",
    "print_cost_and_time",
    "compute_finetuning_cost",
    # Message utilities
    "get_conversation_messages",
    "get_messages",
    "async_get_messages",
    "convert_messages_to_discussion",
    # I/O utilities
    "save_meeting",
    "get_summary",
    "load_summaries",
    "get_recent_markdown",
    "write_final_summary",
    "clear_dir",
    "check_files",
    "load_json",
    "save_json",
]
