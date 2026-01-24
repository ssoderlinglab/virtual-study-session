"""Message conversion and retrieval utilities.

This module provides functions for converting API messages to discussion format
and retrieving messages from conversations.

Usage:
    from virtual_lab.utils.messages import (
        convert_messages_to_discussion,
        get_conversation_messages,
    )

    messages = get_conversation_messages(conversation_id, headers)
    discussion = convert_messages_to_discussion(messages, id_to_title)
"""

from __future__ import annotations

import json
from typing import Any

import requests
from openai import AsyncOpenAI, OpenAI

from virtual_lab.logging_config import get_logger

logger = get_logger(__name__)


def get_conversation_messages(
    conversation_id: str, headers: dict[str, str]
) -> list[dict[str, Any]]:
    """Get messages from an OpenAI conversation via REST API.

    Fetches all messages from a conversation, paginating through results
    as needed.

    Args:
        conversation_id: The ID of the conversation to fetch messages from.
        headers: HTTP headers including authorization.

    Returns:
        A list of message dictionaries.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    messages: list[dict[str, Any]] = []
    last_message: dict[str, Any] | None = None
    url = f"https://api.openai.com/v1/conversations/{conversation_id}/items"
    params: dict[str, Any] = {"order": "asc", "limit": 100}

    while True:
        if last_message is not None:
            params["after"] = last_message["id"]
        elif "after" in params:
            del params["after"]

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = json.loads(response.text).get("data", [])
        new_messages = [msg for msg in data if "content" in msg]

        messages.extend(new_messages)

        if len(new_messages) < params["limit"]:
            break

        if new_messages:
            last_message = messages[-1]

    if not all(len(message.get("content", [])) == 1 for message in messages):
        logger.warning("Some messages have unexpected content structure")

    return messages


def get_messages(client: OpenAI, conversation_id: str) -> list[dict[str, Any]]:
    """Get messages from a conversation using the OpenAI client.

    Args:
        client: The OpenAI client instance.
        conversation_id: The ID of the conversation to get messages from.

    Returns:
        A list of message dictionaries.
    """
    messages: list[dict[str, Any]] = []
    last_message: dict[str, Any] | None = None
    params: dict[str, Any] = {
        "metadata": {"conversation_id": conversation_id},
    }

    while True:
        if last_message is not None:
            params["after"] = last_message["id"]
        elif "after" in params:
            del params["after"]

        new_messages = [
            message.to_dict() for message in client.responses.create(**params)
        ]

        messages.extend(new_messages)

        if len(new_messages) < params.get("limit", 100):
            break

        if messages:
            last_message = messages[-1]

    if not all(len(message.get("content", [])) == 1 for message in messages):
        logger.warning("Some messages have unexpected content structure")

    return messages


async def async_get_messages(
    client: AsyncOpenAI, conversation_id: str
) -> list[dict[str, Any]]:
    """Get messages from a conversation asynchronously.

    Args:
        client: The async OpenAI client instance.
        conversation_id: The ID of the conversation to get messages from.

    Returns:
        A list of message dictionaries.
    """
    messages: list[dict[str, Any]] = []
    last_message: dict[str, Any] | None = None
    params: dict[str, Any] = {
        "metadata": {"conversation_id": conversation_id},
        "limit": 100,
        "order": "asc",
    }

    while True:
        if last_message is not None:
            params["after"] = last_message["id"]
        elif "after" in params:
            del params["after"]

        new_messages = [
            message.to_dict()
            async for message in client.responses.create(**params)
        ]

        messages.extend(new_messages)

        if len(new_messages) < params["limit"]:
            break

        if messages:
            last_message = messages[-1]

    if not all(len(message.get("content", [])) == 1 for message in messages):
        logger.warning("Some messages have unexpected content structure")

    return messages


def convert_messages_to_discussion(
    messages: list[dict[str, Any]],
    assistant_id_to_title: dict[str, str],
) -> list[dict[str, str]]:
    """Convert API conversation items into discussion format.

    Transforms API response messages into a list of dictionaries with
    'agent' and 'message' keys.

    Args:
        messages: List of message dictionaries from the API.
        assistant_id_to_title: Mapping from assistant IDs to agent titles.

    Returns:
        A list of dictionaries with 'agent' and 'message' keys.
    """

    def extract_text(msg: dict[str, Any]) -> str:
        """Extract text content from a message.

        Handles multiple API response formats:
        - Conversations/Responses items with content list
        - Legacy format with nested text.value
        - Direct string content
        """
        content = msg.get("content", [])

        # Primary path: list of parts
        if isinstance(content, list) and content:
            texts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("input_text", "output_text"):
                    txt = part.get("text")
                    if isinstance(txt, str):
                        texts.append(txt)
                    elif isinstance(txt, dict) and isinstance(txt.get("value"), str):
                        texts.append(txt["value"])
            if texts:
                return "\n".join(texts)

        # Legacy fallback: content[0]["text"]["value"]
        try:
            val = msg["content"][0]["text"]["value"]
            if isinstance(val, str):
                return val
        except (KeyError, IndexError, TypeError):
            pass

        # If content itself is a string
        if isinstance(content, str):
            return content

        return ""

    def resolve_agent(msg: dict[str, Any]) -> str:
        """Resolve the agent name from a message."""
        # Prefer explicit assistant id if present and mapped
        a_id = msg.get("assistant_id")
        if a_id and a_id in assistant_id_to_title:
            return assistant_id_to_title[a_id]

        # Some Responses items carry response_id
        r_id = msg.get("response_id")
        if r_id and r_id in assistant_id_to_title:
            return assistant_id_to_title[r_id]

        # Fall back to role-based naming
        role = msg.get("role")
        if role == "user":
            return "User"
        if role == "assistant":
            return "Assistant"

        # Final fallback
        return "Assistant"

    return [{"agent": resolve_agent(m), "message": extract_text(m)} for m in messages]
