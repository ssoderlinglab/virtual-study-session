"""Runtime configuration for Virtual Lab.

This module provides a centralized configuration class that can be initialized
from environment variables, configuration files, or direct instantiation.

Usage:
    from virtual_lab.config.settings import VirtualLabConfig, get_config

    # Get the global configuration
    config = get_config()

    # Or create a custom configuration
    config = VirtualLabConfig(
        api_key="sk-...",
        default_model="gpt-5",
        temperature=0.3,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from virtual_lab.config.constants import (
    DEFAULT_MODEL,
    CONSISTENT_TEMPERATURE,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)


@dataclass
class VirtualLabConfig:
    """Configuration for Virtual Lab runtime.

    This class encapsulates all runtime configuration settings. It can be
    instantiated directly or via the get_config() function which reads
    from environment variables.

    Attributes:
        api_key: OpenAI API key (required for API calls).
        default_model: Default LLM model to use.
        temperature: Default sampling temperature.
        max_retries: Maximum number of API retry attempts.
        retry_base_delay: Base delay in seconds for exponential backoff.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        project_root: Root directory of the project.
    """

    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    default_model: str = DEFAULT_MODEL
    temperature: float = CONSISTENT_TEMPERATURE
    max_retries: int = MAX_RETRIES
    retry_base_delay: float = RETRY_BASE_DELAY
    log_level: str = "INFO"
    log_file: str | None = None
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2: {self.temperature}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative: {self.max_retries}")
        if self.retry_base_delay < 0:
            raise ValueError(
                f"retry_base_delay must be non-negative: {self.retry_base_delay}"
            )

        # Convert project_root to Path if needed
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)

    @property
    def has_api_key(self) -> bool:
        """Check if an API key is configured."""
        return bool(self.api_key)

    def validate(self) -> list[str]:
        """Validate the configuration and return a list of errors.

        Returns:
            A list of error messages. Empty list if valid.
        """
        errors = []
        if not self.has_api_key:
            errors.append("OPENAI_API_KEY is not set")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Note: API key is redacted in the output for security.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "api_key": "***REDACTED***" if self.api_key else "",
            "default_model": self.default_model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "project_root": str(self.project_root),
        }


# Global configuration instance
_config: VirtualLabConfig | None = None


def get_config() -> VirtualLabConfig:
    """Get the global configuration instance.

    Creates the configuration from environment variables on first call,
    then returns the cached instance on subsequent calls.

    Returns:
        The global VirtualLabConfig instance.

    Example:
        config = get_config()
        print(config.default_model)
    """
    global _config
    if _config is None:
        _config = VirtualLabConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            default_model=os.environ.get("VIRTUAL_LAB_MODEL", DEFAULT_MODEL),
            temperature=float(
                os.environ.get("VIRTUAL_LAB_TEMPERATURE", str(CONSISTENT_TEMPERATURE))
            ),
            log_level=os.environ.get("VIRTUAL_LAB_LOG_LEVEL", "INFO"),
            log_file=os.environ.get("VIRTUAL_LAB_LOG_FILE"),
        )
    return _config


def set_config(config: VirtualLabConfig) -> None:
    """Set the global configuration instance.

    This allows overriding the default configuration, which is useful
    for testing or when configuration comes from a non-standard source.

    Args:
        config: The configuration instance to use globally.
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to None.

    The next call to get_config() will create a new configuration from
    environment variables.
    """
    global _config
    _config = None
