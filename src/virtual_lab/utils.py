"""Utility functions for Virtual Lab (backward compatibility shim).

DEPRECATED: This module is maintained for backward compatibility only.
Please import from the specific submodules instead:

    from virtual_lab.utils.tokens import count_tokens, compute_token_cost
    from virtual_lab.utils.messages import convert_messages_to_discussion
    from virtual_lab.utils.io import save_meeting, load_summaries
    from virtual_lab.tools.pubmed import run_pubmed_search

Or use the package import:

    from virtual_lab.utils import count_tokens, save_meeting
"""

import warnings

# Re-export everything from the new modules for backward compatibility
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
)
from virtual_lab.tools.pubmed import (
    get_pubmed_central_article,
    run_pubmed_search,
    run_tools,
)

# Issue deprecation warning on import
warnings.warn(
    "Importing from virtual_lab.utils is deprecated. "
    "Please import from virtual_lab.utils.* or virtual_lab.tools.* instead.",
    DeprecationWarning,
    stacklevel=2,
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
    # PubMed tools (moved to virtual_lab.tools.pubmed)
    "get_pubmed_central_article",
    "run_pubmed_search",
    "run_tools",
]
