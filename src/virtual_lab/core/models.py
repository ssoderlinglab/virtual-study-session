"""Data models for Virtual Lab.

This module provides structured data classes for configuration and results,
ensuring type safety and clear interfaces throughout the codebase.

Usage:
    from virtual_lab.core.models import MeetingConfig, MeetingResult

    config = MeetingConfig(
        meeting_type="team",
        agenda="Review grant significance",
        num_rounds=2,
    )

    result = MeetingResult(
        discussion=[{"agent": "Chair", "message": "..."}],
        conversation_id="conv_123",
        token_counts={"input": 1000, "output": 500},
        cost_usd=0.05,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class MeetingConfig:
    """Configuration for a meeting.

    This dataclass encapsulates all the configuration parameters needed to
    run a meeting, providing type-safe access to meeting settings.

    Attributes:
        meeting_type: The type of meeting ("team" or "individual").
        agenda: The agenda for the meeting.
        num_rounds: Number of discussion rounds (default 0 for single round).
        temperature: LLM sampling temperature (default 0.2).
        save_dir: Directory to save meeting outputs.
        save_name: Base name for saved files (default "discussion").
        pubmed_search: Whether to enable PubMed search tool.
        return_summary: Whether to return the meeting summary.
    """

    meeting_type: Literal["team", "individual"]
    agenda: str
    num_rounds: int = 0
    temperature: float = 0.2
    save_dir: Path | None = None
    save_name: str = "discussion"
    pubmed_search: bool = False
    return_summary: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.meeting_type not in ("team", "individual"):
            raise ValueError(f"Invalid meeting type: {self.meeting_type}")
        if self.num_rounds < 0:
            raise ValueError(f"num_rounds must be non-negative: {self.num_rounds}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2: {self.temperature}")


@dataclass
class MeetingResult:
    """Result of a completed meeting.

    This dataclass encapsulates all the outputs from a meeting, including
    the full discussion, token usage, and cost information.

    Attributes:
        discussion: List of message dicts with 'agent' and 'message' keys.
        conversation_id: The OpenAI conversation ID.
        token_counts: Dictionary of token counts (input, output, tool, max).
        cost_usd: Estimated cost in USD.
        summary: The final summary message (if return_summary was True).
        save_paths: Dictionary of paths to saved files (json, md).
    """

    discussion: list[dict[str, str]]
    conversation_id: str
    token_counts: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    summary: str | None = None
    save_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        """Return the number of messages in the discussion."""
        return len(self.discussion)

    @property
    def total_tokens(self) -> int:
        """Return the total token count (input + output + tool)."""
        return (
            self.token_counts.get("input", 0)
            + self.token_counts.get("output", 0)
            + self.token_counts.get("tool", 0)
        )


@dataclass
class AgentConfig:
    """Configuration for creating an Agent.

    This dataclass provides a structured way to define agent parameters,
    separate from the Agent class itself.

    Attributes:
        title: The title of the agent (e.g., "Primary Reviewer").
        expertise: Description of the agent's expertise.
        goal: The agent's goal in the context of the project.
        role: The agent's role in meetings.
        model: The LLM model to use (default "gpt-5-mini").
    """

    title: str
    expertise: str
    goal: str
    role: str
    model: str = "gpt-5-mini"


@dataclass
class WorkflowConfig:
    """Configuration for a workflow run.

    This dataclass encapsulates all the settings needed to run a complete
    workflow, including paths, phases, and runtime options.

    Attributes:
        grant_filepath: Path to the grant file to review.
        output_dir: Base directory for output files.
        phases: List of phase names to execute.
        num_iterations: Number of iterations per phase.
        num_rounds: Number of discussion rounds per meeting.
        model: Default LLM model to use.
        clear_dirs: Whether to clear output directories before running.
        short_mode: Whether to run in abbreviated "short" mode.
    """

    grant_filepath: Path
    output_dir: Path
    phases: list[str] = field(default_factory=list)
    num_iterations: int = 1
    num_rounds: int = 2
    model: str = "gpt-5-mini"
    clear_dirs: bool = False
    short_mode: bool = False

    def __post_init__(self) -> None:
        """Validate and convert paths after initialization."""
        if isinstance(self.grant_filepath, str):
            self.grant_filepath = Path(self.grant_filepath)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if not self.phases:
            self.phases = [
                "team_selection",
                "independent_review",
                "collaboration_review",
                "chair_merge",
                "final_output",
            ]


@dataclass
class TokenUsage:
    """Token usage statistics for a meeting or workflow.

    Attributes:
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        tool_tokens: Number of tokens used by tool calls.
        max_context: Maximum context length reached.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    tool_tokens: int = 0
    max_context: int = 0

    @property
    def total(self) -> int:
        """Return total tokens (input + output + tool)."""
        return self.input_tokens + self.output_tokens + self.tool_tokens

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary format used by existing code."""
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "tool": self.tool_tokens,
            "max": self.max_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "TokenUsage":
        """Create from dictionary format used by existing code."""
        return cls(
            input_tokens=data.get("input", 0),
            output_tokens=data.get("output", 0),
            tool_tokens=data.get("tool", 0),
            max_context=data.get("max", 0),
        )


@dataclass
class ScoreResult:
    """NIH-style score result for a single criterion.

    Attributes:
        criterion: The criterion being scored (e.g., "Significance").
        score: Integer score from 1-9.
        justification: Explanation for the score.
    """

    criterion: str
    score: int
    justification: str

    def __post_init__(self) -> None:
        """Validate score range."""
        if not 1 <= self.score <= 9:
            raise ValueError(f"Score must be between 1 and 9: {self.score}")


@dataclass
class ReviewResult:
    """Complete review result with all scores and summary.

    Attributes:
        scores: List of ScoreResult for each criterion.
        summary_statement: Overall summary statement.
        reviewer_id: Identifier for the reviewer.
        aim: The aim being reviewed (if applicable).
    """

    scores: list[ScoreResult]
    summary_statement: str
    reviewer_id: str | None = None
    aim: str | None = None

    @property
    def overall_score(self) -> int | None:
        """Return the Overall Impact score if present."""
        for score in self.scores:
            if score.criterion.lower() == "overall impact":
                return score.score
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "scores": {
                s.criterion: {"score": s.score, "justification": s.justification}
                for s in self.scores
            },
            "summary_statement": self.summary_statement,
            "reviewer_id": self.reviewer_id,
            "aim": self.aim,
        }
