"""Custom exception hierarchy for Virtual Lab.

This module defines a structured exception hierarchy that allows for precise
error handling and informative error messages throughout the codebase.

Usage:
    from virtual_lab.core.exceptions import APIError, RateLimitError

    try:
        response = api_call()
    except RateLimitError as e:
        logger.warning(f"Rate limited, retrying: {e}")
        time.sleep(e.retry_after)
    except APIError as e:
        logger.error(f"API error: {e}")
"""

from __future__ import annotations

from typing import Any


class VirtualLabError(Exception):
    """Base exception for all Virtual Lab errors.

    All custom exceptions in the Virtual Lab should inherit from this class
    to allow catching all Virtual Lab-specific errors with a single except clause.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation including details if present."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class APIError(VirtualLabError):
    """Exception raised for OpenAI API errors.

    This exception is raised when an API call fails for reasons other than
    rate limiting (which has its own exception).

    Attributes:
        status_code: HTTP status code from the API response.
        response_body: Raw response body from the API.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the API error.

        Args:
            message: Human-readable error description.
            status_code: HTTP status code from the API response.
            response_body: Raw response body from the API.
            details: Optional dictionary with additional error context.
        """
        details = details or {}
        if status_code is not None:
            details["status_code"] = status_code
        if response_body is not None:
            details["response_body"] = response_body[:500]  # Truncate long responses

        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(APIError):
    """Exception raised when API rate limits are exceeded.

    This exception includes retry timing information when available.

    Attributes:
        retry_after: Suggested seconds to wait before retrying (from API headers).
    """

    def __init__(
        self,
        message: str = "API rate limit exceeded",
        retry_after: float | None = None,
        status_code: int = 429,
        response_body: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the rate limit error.

        Args:
            message: Human-readable error description.
            retry_after: Suggested seconds to wait before retrying.
            status_code: HTTP status code (defaults to 429).
            response_body: Raw response body from the API.
            details: Optional dictionary with additional error context.
        """
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after

        super().__init__(message, status_code, response_body, details)
        self.retry_after = retry_after


class MeetingError(VirtualLabError):
    """Exception raised for errors during meeting execution.

    This exception is raised when a meeting fails to complete successfully,
    whether due to agent errors, conversation issues, or other meeting-specific
    problems.

    Attributes:
        meeting_type: The type of meeting that failed ("team" or "individual").
        phase: The phase of the meeting where the error occurred.
        conversation_id: The conversation ID if available.
    """

    def __init__(
        self,
        message: str,
        meeting_type: str | None = None,
        phase: str | None = None,
        conversation_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the meeting error.

        Args:
            message: Human-readable error description.
            meeting_type: The type of meeting that failed.
            phase: The phase of the meeting where the error occurred.
            conversation_id: The conversation ID if available.
            details: Optional dictionary with additional error context.
        """
        details = details or {}
        if meeting_type is not None:
            details["meeting_type"] = meeting_type
        if phase is not None:
            details["phase"] = phase
        if conversation_id is not None:
            details["conversation_id"] = conversation_id

        super().__init__(message, details)
        self.meeting_type = meeting_type
        self.phase = phase
        self.conversation_id = conversation_id


class ConfigurationError(VirtualLabError):
    """Exception raised for configuration errors.

    This exception is raised when configuration is missing, invalid, or
    inconsistent.

    Attributes:
        config_key: The configuration key that caused the error.
        expected_type: The expected type of the configuration value.
        actual_value: The actual value that was found (if any).
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        expected_type: type | None = None,
        actual_value: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the configuration error.

        Args:
            message: Human-readable error description.
            config_key: The configuration key that caused the error.
            expected_type: The expected type of the configuration value.
            actual_value: The actual value that was found.
            details: Optional dictionary with additional error context.
        """
        details = details or {}
        if config_key is not None:
            details["config_key"] = config_key
        if expected_type is not None:
            details["expected_type"] = expected_type.__name__
        if actual_value is not None:
            details["actual_value"] = repr(actual_value)[:100]

        super().__init__(message, details)
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class ParsingError(VirtualLabError):
    """Exception raised for response parsing errors.

    This exception is raised when an API response or file cannot be parsed
    as expected.

    Attributes:
        source: The source of the data that failed to parse.
        expected_format: Description of the expected format.
        raw_content: The raw content that failed to parse (truncated).
    """

    def __init__(
        self,
        message: str,
        source: str | None = None,
        expected_format: str | None = None,
        raw_content: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the parsing error.

        Args:
            message: Human-readable error description.
            source: The source of the data that failed to parse.
            expected_format: Description of the expected format.
            raw_content: The raw content that failed to parse.
            details: Optional dictionary with additional error context.
        """
        details = details or {}
        if source is not None:
            details["source"] = source
        if expected_format is not None:
            details["expected_format"] = expected_format
        if raw_content is not None:
            details["raw_content_preview"] = raw_content[:200]

        super().__init__(message, details)
        self.source = source
        self.expected_format = expected_format
        self.raw_content = raw_content
