"""Runs a meeting with LLM agents using the Responses API."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal, Optional

import requests
from tqdm import trange, tqdm

from virtual_lab.agent import Agent
from virtual_lab.config.constants import CONSISTENT_TEMPERATURE, PUBMED_TOOL_DESCRIPTION
from virtual_lab.core.exceptions import APIError, MeetingError, RateLimitError
from virtual_lab.logging_config import get_logger
from virtual_lab.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
)
from virtual_lab.utils.messages import (
    convert_messages_to_discussion,
    get_conversation_messages,
)
from virtual_lab.utils.tokens import (
    count_discussion_tokens,
    print_cost_and_time,
)
from virtual_lab.utils.io import (
    get_summary,
    save_meeting,
)

logger = get_logger(__name__)


def create_conversation(headers: dict[str, str]) -> dict[str, Any]:
    """Create a new OpenAI conversation.

    Args:
        headers: HTTP headers including authorization.

    Returns:
        The conversation JSON response.

    Raises:
        APIError: If the conversation cannot be created.
    """
    url = "https://api.openai.com/v1/conversations"
    payload = {
        "items": [
            {
                "type": "message",
                "role": "user",
                "content": "Hello!"
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        error_body = ""
        try:
            error_body = json.dumps(response.json(), indent=2)
        except Exception:
            error_body = response.text[:1000]
        logger.error(f"Failed to create conversation: {error_body}")
        raise APIError(
            f"Failed to create conversation: {e}",
            status_code=response.status_code,
            response_body=error_body,
        ) from e
    except requests.RequestException as e:
        logger.error(f"Request error creating conversation: {e}")
        raise APIError(f"Request error creating conversation: {e}") from e

def post_response(
    payload: dict[str, Any],
    base_url: str,
    headers: dict[str, str],
    meeting_type: str,
    meeting_genre: Path,
    round_idx: int,
) -> dict[str, Any]:
    """Post a response request to the OpenAI API.

    Args:
        payload: The request payload.
        base_url: The base API URL.
        headers: HTTP headers including authorization.
        meeting_type: Type of meeting for logging.
        meeting_genre: Meeting directory for logging.
        round_idx: Current round index for logging.

    Returns:
        The API response JSON.

    Raises:
        APIError: If the request fails after retries.
    """
    responses_url = f"{base_url}/responses"

    response = post_with_retries(
        responses_url,
        headers=headers,
        payload=payload,
        meeting_type=meeting_type,
        meeting_genre=meeting_genre,
        round_idx=round_idx,
    )

    try:
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        error_body = ""
        try:
            error_body = json.dumps(response.json(), indent=2)
            logger.error(f"API error response: {error_body}")
        except Exception:
            error_body = response.text[:1000]
            logger.error(f"API error (raw): {error_body}")

        if response.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded",
                status_code=response.status_code,
                response_body=error_body,
            ) from e

        raise APIError(
            f"API request failed: {e}",
            status_code=response.status_code,
            response_body=error_body,
        ) from e


def post_with_retries(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    meeting_type: str,
    meeting_genre: Path,
    round_idx: int,
    max_tries: int = 6,
    base_delay: float = 10.0,
) -> requests.Response:
    """Post a request with exponential backoff retries.

    Args:
        url: The URL to post to.
        headers: HTTP headers.
        payload: Request payload.
        meeting_type: Type of meeting for logging.
        meeting_genre: Meeting directory for logging.
        round_idx: Current round index for logging.
        max_tries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.

    Returns:
        The response object.
    """
    last_response = None

    for attempt in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)

            # Success or non-retryable error
            if response.status_code < 500 and response.status_code not in (408, 429):
                return response

            last_response = response
            logger.warning(
                f"Retryable error on {meeting_type} {meeting_genre.stem} "
                f"round {round_idx}: status {response.status_code} "
                f"(attempt {attempt + 1}/{max_tries})"
            )

        except requests.RequestException as e:
            logger.warning(
                f"Request exception on {meeting_type} {meeting_genre.stem} "
                f"round {round_idx}: {e} (attempt {attempt + 1}/{max_tries})"
            )

        # Wait before retry (exponential backoff would be better)
        if attempt < max_tries - 1:
            time.sleep(base_delay)

    if last_response is not None:
        return last_response

    # Should not reach here, but return empty response as fallback
    raise APIError(f"All {max_tries} retry attempts failed")


def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    pubmed_search: bool = False,
    return_summary: bool = False,
    OVERALL_MODEL: str = 'gpt-5',
    OVERALL_MINI_MODEL: str= 'gpt-5 mini',
    conversation_id: Optional[str]=None
) -> tuple[str | tuple[()], str]:
    """Runs a meeting with LLM agents using the Responses API.

    :param meeting_type: The type of meeting.
    :param agenda: The agenda for the meeting.
    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param team_lead: The team lead for a team meeting (None for individual meeting).
    :param team_members: The team members for a team meeting (None for individual meeting).
    :param team_member: The team member for an individual meeting (None for team meeting).
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the meeting.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param num_rounds: The number of rounds of discussion.
    :param temperature: The sampling temperature.
    :param pubmed_search: Whether to include a PubMed search tool.
    :param return_summary: Whether to return the summary of the meeting.
    :return: The summary of the meeting if return_summary is True, else None.
    """
    # Validate meeting type
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_member is not None:
            raise ValueError("Team meeting does not require individual team member")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        if team_lead is not None or team_members is not None:
            raise ValueError(
                "Individual meeting does not require team lead or team members"
            )
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    # Start timing the meeting
    start_time = time.time()

    ## start api call
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")
    BASE = "https://api.openai.com/v1"
    RESPONSES = f"{BASE}/responses"
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # start conversation
    if not conversation_id:
        conv_json = create_conversation(HEADERS)
        conversation_id = conv_json['id']  # depending on SDK version

    # Set up team
    if meeting_type == "team":
        team = [team_lead] + list(team_members)
    else:
        team = [team_member] + [SCIENTIFIC_CRITIC]
    # Set up tools
    assistant_params = {"tools": [PUBMED_TOOL_DESCRIPTION]} if pubmed_search else {}
    
    # Set up the assistants
    agent_to_assistant = {}
    for agent in team:
        payload = {'input': agent.prompt,
                  'model': agent.model,
                  'conversation': conversation_id,
                  'truncation': 'auto'}
        r_json = post_response(payload, base_url=BASE, headers=HEADERS, meeting_type=meeting_type, meeting_genre=save_dir, round_idx=0)
        agent_to_assistant[agent] = r_json ## need to return the message only?


    # Map assistant IDs to agents
    assistant_id_to_title = {
        assistant['id']: agent.title for agent, assistant in agent_to_assistant.items()
    }
    

    # Set up tool token count
    tool_token_count = 0

    if meeting_type == "team":
        payload = {'model': OVERALL_MODEL,
                   'conversation': conversation_id,
                   'truncation':'auto',
                  'input' : team_meeting_start_prompt(
                team_lead=team_lead,
                team_members=team_members,
                agenda=agenda,
                agenda_questions=agenda_questions,
                agenda_rules=agenda_rules,
                summaries=summaries,
                contexts=contexts,
                num_rounds=num_rounds,
            )}

    # Loop through rounds
    for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
        round_num = round_index + 1

        # Loop through team and elicit responses
        for agent in tqdm(team, desc="Team"):
            time.sleep(5)
            # Prompt based on agent and round number
            if meeting_type == "team":
                # Team meeting prompts
                if agent == team_lead:
                    if round_index == 0:
                        prompt = team_meeting_team_lead_initial_prompt(
                            team_lead=team_lead
                        )
                    elif round_index == num_rounds:
                        prompt = team_meeting_team_lead_final_prompt(
                            team_lead=team_lead,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                        )
                    else:
                        prompt = team_meeting_team_lead_intermediate_prompt(
                            team_lead=team_lead,
                            round_num=round_num - 1,
                            num_rounds=num_rounds,
                        )
                else:
                    prompt = team_meeting_team_member_prompt(
                        team_member=agent, round_num=round_num, num_rounds=num_rounds
                    )
            else:
                # Individual meeting prompts
                if agent == SCIENTIFIC_CRITIC:
                    prompt = individual_meeting_critic_prompt(
                        critic=SCIENTIFIC_CRITIC, agent=team_member
                    )
                else:
                    if round_index == 0:
                        prompt = individual_meeting_start_prompt(
                            team_member=team_member,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                            summaries=summaries,
                            contexts=contexts,
                        )
                    else:
                        prompt = individual_meeting_agent_prompt(
                            critic=SCIENTIFIC_CRITIC, agent=team_member
                        )
            
            # run agent
            payload = {'input': prompt,
                      'model': OVERALL_MODEL,
                      # 'previous_response_id': agent_to_assistant[agent]['id'],
                      'conversation': conversation_id, 
                      'truncation': 'auto'}
            run = post_response(
                payload,
                base_url=BASE,
                headers=HEADERS,
                meeting_type=meeting_type,
                meeting_genre=save_dir,
                round_idx=round_num,
            )

            # Handle tool calls (PubMed search)
            if run.get('status') == 'requires_action':
                logger.warning(
                    "Tool action required but not implemented for gpt-5. "
                    "PubMed tool support requires Assistants API."
                )
                # TODO: Implement tool handling for Responses API when available

            # Check for run failure
            if run.get('status') != "completed":
                run_status = run.get('status', 'unknown')
                run_id = run.get('id', 'unknown')
                last_error = run.get('last_error')
                error_details = run.get('error')

                logger.error(
                    f"Run failed: status={run_status}, id={run_id}, "
                    f"last_error={last_error}, error={error_details}"
                )

                raise MeetingError(
                    f"Run failed with status: {run_status}",
                    meeting_type=meeting_type,
                    phase=f"round_{round_num}",
                    conversation_id=conversation_id,
                    details={
                        "run_id": run_id,
                        "status": run_status,
                        "last_error": last_error,
                        "error": error_details,
                    },
                )
            # If final round, only team lead or team member responds
            if round_index == num_rounds:
                break
    
    # Get messages from the discussion
    messages = get_conversation_messages(conversation_id=conversation_id, headers=HEADERS)

    # Convert messages to discussion format
    discussion = convert_messages_to_discussion(
        messages=messages, assistant_id_to_title=assistant_id_to_title
    )

    # Count discussion tokens
    token_counts = count_discussion_tokens(discussion=discussion)

    # Add tool token count to total token count
    token_counts["tool"] = tool_token_count

    # Print cost and time
    # TODO: handle different models for different agents
    print_cost_and_time(
        token_counts=token_counts,
        model=team_lead.model if meeting_type == "team" else team_member.model,
        elapsed_time=time.time() - start_time,
    )

    # Save the discussion as JSON and Markdown
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion,
    )

    # Optionally, return summary
    if return_summary:
        return get_summary(discussion), conversation_id
    return (), conversation_id
