''"""Runs a meeting with LLM agents using the Responses API."""

import time
from pathlib import Path
from typing import Literal, Optional

import openai
from openai import OpenAI 
from tqdm import trange, tqdm

from virtual_lab.agent import Agent
from virtual_lab.constants import CONSISTENT_TEMPERATURE, PUBMED_TOOL_DESCRIPTION
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
from virtual_lab.utils import (
    convert_messages_to_discussion,
    count_discussion_tokens,
    count_tokens,
    get_messages,
    get_summary,
    print_cost_and_time,
    run_tools,
    save_meeting,
    get_conversation_messages
)

from pprint import pprint
from uuid import uuid4
import os, requests, json,time
import time, random, requests


def create_conversation( headers):
    w = f'https://api.openai.com/v1/conversations'
    payload = {'items': [
      {
        "type": "message",
        "role": "user",
        "content": "Hello!"
      }
    ]}
    r = requests.post(w, headers=headers, json=payload)
    try:
         r.raise_for_status()
    except requests.HTTPError:
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text[:1000])
    return r.json()

def post_response(payload: dict, BASE, HEADERS, meeting_type, meeting_genre, round_idx) -> dict:
    RESPONSES =  f'{BASE}/responses' 
    ## log probs is a param parameter.. 
    params = {'include': ''}
    
#     r = requests.post(RESPONSES, headers=HEADERS,
#                   json=payload)
    r = post_with_retries(RESPONSES, headers=HEADERS, payload=payload, meeting_type=meeting_type, meeting_genre = meeting_genre, round_idx = round_idx)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print('status: ', r.status_code)
        try:
            p =  r.json()
            print('failing json: ', p)
        except Exception:
            print('failed to do json dump')
            print(r.text[:1000])
        raise
    return r.json()

def post_with_retries(responses, headers, payload, meeting_type, meeting_genre, round_idx, max_tries=6, base=0.6):
    for i in range(max_tries):
        r = requests.post(responses, headers=headers,
              json=payload)
        if r.status_code < 500 and r.status_code not in (408, 429):
            return r
        time.sleep(10)
        print(f'{meeting_type}, {meeting_genre.stem} on round {round_idx} failed')
    return r


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
) -> str:
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
        r_json = post_response(payload, BASE=BASE, HEADERS=HEADERS, meeting_type=meeting_type, meeting_genre=save_dir, round_idx=0)
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
            run = post_response(payload, BASE=BASE, HEADERS=HEADERS, meeting_type=meeting_type, meeting_genre=save_dir, round_idx=round_num)
            if run['status'] == 'requires_action':
                print('a TOOL  is BEING USED, setup not completed for pubmed tool with gpt-5. This is'
                     ' still on assistants api, so client.beta.threads.runs.submit_tool_outputs_and_poll')
#                 tool_outputs = run_tools(run=run)
                
#                                 # Update tool token count
#                 tool_token_count += sum(
#                     count_tokens(tool_output["output"]) for tool_output in tool_outputs
#                 )

#                                 # Submit the tool outputs
#                 run = client.beta.threads.runs.submit_tool_outputs_and_poll(
#                     thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
#                 )

#                 # Add tool outputs to the thread so it's visible for later rounds
#                 client.beta.threads.messages.create(
#                     thread_id=thread.id,
#                     role="user",
#                     content="Tool Output:\n\n"
#                     + "\n\n".join(
#                         tool_output["output"] for tool_output in tool_outputs
#                     ),
#                 )
            
            if run['status'] != "completed":
                #### RUN IS NOW A DICTIONARY-- NO ATTRIBUTES
                print("[run] status:", getattr(run, "status", None))
                print("[run] id:", getattr(run, "id", None))
                print("[run] model:", getattr(run, "model", None))
                print("[run] usage:", getattr(run, "usage", None))

                # most APIs expose one or more of these on failure:
                print("[run] last_error:", getattr(run, "last_error", None))
                print("[run] error:", getattr(run, "error", None))
                print("[run] incomplete_details:", getattr(run, "incomplete_details", None))
                print("[run] required_action:", getattr(run, "required_action", None))

                # tails (if present)
                tail_msgs = getattr(run, "output_messages", None)
                if tail_msgs is not None:
                    print("[run] output_messages tail:")
                    pprint(tail_msgs[-3:])

                tail_tools = getattr(run, "tool_outputs", None)
                if tail_tools is not None:
                    print("[run] tool_outputs tail:")
                    pprint(tail_tools[-3:])

                # as a catch-all: shows any other attrs if it's a normal object
                try:
                    print("[run] __dict__ keys:", list(vars(run).keys()))
                except TypeError:
                    pass
                print(f'run failed at, {run.failed_at}\n\nLast error: {run.last_error}')
                raise ValueError(f"Run failed: {run.status}")
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
