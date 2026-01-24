"""File I/O utilities for Virtual Lab.

This module provides functions for saving and loading meeting discussions,
summaries, and other file operations.

Usage:
    from virtual_lab.utils.io import save_meeting, load_summaries, get_summary

    save_meeting(save_dir, "discussion", discussion)
    summaries = load_summaries(discussion_paths)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from virtual_lab.logging_config import get_logger

logger = get_logger(__name__)


def save_meeting(
    save_dir: Path, save_name: str, discussion: list[dict[str, str]]
) -> dict[str, Path]:
    """Save a meeting discussion to JSON and Markdown files.

    Creates the save directory if it doesn't exist, then writes the discussion
    as both structured JSON and human-readable Markdown.

    Args:
        save_dir: The directory to save the discussion.
        save_name: The base name for the saved files (without extension).
        discussion: The discussion to save, as a list of dicts with
            'agent' and 'message' keys.

    Returns:
        Dictionary mapping format names ('json', 'md') to their file paths.

    Example:
        paths = save_meeting(
            Path("discussions/phase1"),
            "discussion_1",
            [{"agent": "Chair", "message": "Welcome"}]
        )
        # paths = {'json': Path(...), 'md': Path(...)}
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    json_path = save_dir / f"{save_name}.json"
    md_path = save_dir / f"{save_name}.md"

    # Save as JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(discussion, f, indent=4)
    logger.debug(f"Saved JSON to {json_path}")

    # Save as Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        for turn in discussion:
            f.write(f"## {turn['agent']}\n\n{turn['message']}\n\n")
    logger.debug(f"Saved Markdown to {md_path}")

    return {"json": json_path, "md": md_path}


def get_summary(discussion: list[dict[str, str]]) -> str:
    """Extract the summary from a discussion.

    The summary is conventionally the last message in the discussion,
    typically from the team lead or the agent wrapping up.

    Args:
        discussion: The discussion to extract the summary from.

    Returns:
        The text of the last message in the discussion.

    Raises:
        IndexError: If the discussion is empty.
    """
    if not discussion:
        raise IndexError("Cannot get summary from empty discussion")
    return discussion[-1]["message"]


def load_summaries(discussion_paths: list[Path]) -> tuple[str, ...]:
    """Load summaries from a list of discussion JSON files.

    Each discussion file is expected to be a JSON array of message dicts,
    and the summary is extracted as the last message.

    Args:
        discussion_paths: Paths to the discussion JSON files.

    Returns:
        A tuple of summary strings, one per discussion file.

    Example:
        summaries = load_summaries([
            Path("discussions/discussion_1.json"),
            Path("discussions/discussion_2.json"),
        ])
    """
    summaries = []
    for discussion_path in discussion_paths:
        try:
            with open(discussion_path, "r", encoding="utf-8") as f:
                discussion = json.load(f)
            summaries.append(get_summary(discussion))
        except (json.JSONDecodeError, FileNotFoundError, IndexError) as e:
            logger.warning(f"Failed to load summary from {discussion_path}: {e}")
            continue
    return tuple(summaries)


def get_recent_markdown(path: Path) -> str:
    """Get the content of the most recent Markdown file in a directory.

    Files are sorted alphabetically, so naming conventions like
    "discussion_1.md", "discussion_2.md" will result in the highest
    numbered file being returned.

    Args:
        path: Directory path to search for Markdown files.

    Returns:
        The content of the most recent Markdown file.

    Raises:
        FileNotFoundError: If no Markdown files are found.
    """
    md_files = sorted(path.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No Markdown files found in {path}")

    md_file = md_files[-1]
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    logger.debug(f"Loaded recent Markdown from {md_file}")
    return content


def write_final_summary(
    save_dir: Path, summary: str, grant_name: str, out_suffix: str
) -> Path:
    """Write a final summary to a text file.

    Args:
        save_dir: Directory to save the summary.
        summary: The summary text to write.
        grant_name: Name of the grant for the filename.
        out_suffix: Suffix for the filename.

    Returns:
        Path to the written file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{grant_name}_{out_suffix}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info(f"Wrote final summary to {output_path}")
    return output_path


def clear_dir(save_dir: Path) -> None:
    """Clear a directory by removing it and all its contents.

    Args:
        save_dir: Directory to clear.
    """
    if save_dir.exists():
        shutil.rmtree(save_dir)
        logger.info(f"Cleared directory {save_dir}")


def check_files(path: Path, min_count: int = 1, pattern: str = "*.json") -> bool:
    """Check if a directory contains at least a minimum number of matching files.

    Args:
        path: Directory path to check.
        min_count: Minimum number of files required.
        pattern: Glob pattern for matching files.

    Returns:
        True if the directory contains at least min_count matching files.
    """
    if not path.exists():
        return False
    files = list(path.glob(pattern))
    return len(files) >= min_count


def load_json(path: Path) -> dict | list:
    """Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict | list, indent: int = 4) -> None:
    """Save data to a JSON file.

    Args:
        path: Path to save the JSON file.
        data: Data to serialize.
        indent: Indentation level for formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    logger.debug(f"Saved JSON to {path}")
