"""Virtual Lab package.

The Virtual Lab is an AI-human collaboration framework for scientific research,
designed to orchestrate multi-agent discussions for tasks like grant review.

Main components:
    - Agent: LLM persona with defined expertise, goals, and role
    - run_meeting: Function to orchestrate agent conversations
    - core: Exceptions, models, and data structures
    - config: Configuration and constants
    - utils: Utility functions for tokens, messages, and I/O
    - tools: External tool integrations (PubMed search)
    - logging_config: Structured logging

Usage:
    from virtual_lab import Agent, run_meeting
    from virtual_lab.core import MeetingConfig, VirtualLabError
    from virtual_lab.config import get_config
"""

from virtual_lab.__about__ import __version__
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting

# Explicit imports for commonly used items
from virtual_lab.core import (
    VirtualLabError,
    APIError,
    RateLimitError,
    MeetingError,
    ConfigurationError,
    ParsingError,
    MeetingConfig,
    MeetingResult,
)
from virtual_lab.config import (
    VirtualLabConfig,
    get_config,
    CONSISTENT_TEMPERATURE,
    CREATIVE_TEMPERATURE,
)
from virtual_lab.logging_config import get_logger, setup_logging


__all__ = [
    # Version
    "__version__",
    # Core classes
    "Agent",
    "run_meeting",
    # Exceptions
    "VirtualLabError",
    "APIError",
    "RateLimitError",
    "MeetingError",
    "ConfigurationError",
    "ParsingError",
    # Models
    "MeetingConfig",
    "MeetingResult",
    # Config
    "VirtualLabConfig",
    "get_config",
    "CONSISTENT_TEMPERATURE",
    "CREATIVE_TEMPERATURE",
    # Logging
    "get_logger",
    "setup_logging",
]
