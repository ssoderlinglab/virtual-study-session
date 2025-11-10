"""Contains useful utility functions."""

import json
import urllib.parse
from pathlib import Path

import requests
import tiktoken
from openai import OpenAI
from openai import AsyncOpenAI, OpenAI
from openai.types.beta.threads.run import Run
from typing import List, Dict, Optional

from virtual_lab.constants import (
    DEFAULT_FINETUNING_EPOCHS,
    MODEL_TO_INPUT_PRICE_PER_TOKEN,
    MODEL_TO_OUTPUT_PRICE_PER_TOKEN,
    FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN,
    PUBMED_TOOL_NAME,
)
from virtual_lab.prompts import format_references
import pathlib 
import shutil

def get_pubmed_central_article(
    pmcid: str, abstract_only: bool = False
) -> tuple[str | None, list[str] | None]:
    """Gets the title and content (abstract or full text) of a PubMed Central article given a PMC ID.

    Note: This only returns main text, ignoring tables, figures, and references.

    :param pmcid: The PMC ID of the article.
    :param abstract_only: Whether to return only the abstract instead of the full text.
    :return: The title and content (abstract or full text of the article as a list of paragraphs)
        or None if the article is not found.
    """
    # Get article from PMC ID in JSON form
    text_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_JSON/PMC{pmcid}/unicode"
    response = requests.get(text_url)
    response.raise_for_status()

    # Try to parse JSON
    try:
        article = response.json()
    except json.JSONDecodeError:
        return None, None

    # Get document
    document = article[0]["documents"][0]

    # Get title
    title = next(
        passage["text"]
        for passage in document["passages"]
        if passage["infons"]["section_type"] == "TITLE"
    )

    # Get relevant passages
    passages = [
        passage
        for passage in document["passages"]
        if passage["infons"]["type"] in {"abstract", "paragraph"}
    ]

    # Get abstract or full text of article (excluding references)
    if abstract_only:
        passages = [
            passage
            for passage in passages
            if passage["infons"]["section_type"] in ["ABSTRACT"]
        ]
    else:
        passages = [
            passage
            for passage in passages
            if passage["infons"]["section_type"]
            in ["ABSTRACT", "INTRO", "RESULTS", "DISCUSS", "CONCL", "METHODS"]
        ]

    # Get content
    content = [passage["text"] for passage in passages]

    return title, content


def run_pubmed_search(
    query: str, num_articles: int = 3, abstract_only: bool = False
) -> str:
    """Runs a PubMed search, returning the full text of the top matching article.

    :param query: The query to search PubMed with.
    :param num_articles: The number of articles to search for.
    :param abstract_only: Whether to return only the abstract instead of the full text.
    :return: The full text of the top matching article.
    """
    # Print search query
    print(
        f"Searching PubMed Central for {num_articles} articles "
        f"({'abstracts' if abstract_only else 'full text'}) with query: \"{query}\""
    )

    # Perform PubMed Central search for query to get PMC ID
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={urllib.parse.quote_plus(query)}&retmax={2 * num_articles}&retmode=json&sort=relevance"
    response = requests.get(search_url)
    response.raise_for_status()
    pmcids_found = response.json()["esearchresult"]["idlist"]

    # Loop through top articles
    texts = []
    titles = []
    pmcids = []

    for pmcid in pmcids_found:
        # Break if reached desired number of articles
        if len(pmcids) >= num_articles:
            break

        title, content = get_pubmed_central_article(
            pmcid=pmcid,
            abstract_only=abstract_only,
        )

        if title is None:
            continue

        texts.append(f"PMCID = {pmcid}\n\nTitle = {title}\n\n{'\n\n'.join(content)}")
        titles.append(title)
        pmcids.append(pmcid)

    # Print articles found
    article_count = len(texts)

    print(f"Found {article_count:,} articles on PubMed Central")

    # Combine texts
    if article_count == 0:
        combined_text = f'No articles found on PubMed Central for the query "{query}".'
    else:
        combined_text = format_references(
            references=tuple(texts),
            reference_type="paper",
            intro=f'Here are the top {article_count} articles on PubMed Central for the query "{query}":',
        )

    return combined_text


def run_tools(run: Run) -> list[dict[str, str]]:
    """Runs the tools in a required action.

    :param run: The run to run tools for.
    :return: A list of tool outputs.
    """
    # Define the list to store tool outputs
    tool_outputs = []

    # Loop through each tool in the required action and run it
    for tool in run.required_action.submit_tool_outputs.tool_calls:
        if tool.function.name == PUBMED_TOOL_NAME:
            # Extract the query from the tool arguments
            args_dict = json.loads(tool.function.arguments)

            # Run the tool and append the output to the list of tool outputs
            tool_outputs.append(
                {
                    "tool_call_id": tool.id,
                    "output": run_pubmed_search(**args_dict),
                }
            )
        else:
            raise ValueError(f"Unknown tool: {tool.function.name}")

    return tool_outputs

def get_conversation_messages(conversation_id, headers):
    '''
    attributes of returned api object
    'apparent_encoding', 'close', 'connection', 'content', 'cookies', 'elapsed', 
    'encoding', 'headers', 'history', 'is_permanent_redirect', 'is_redirect', 
    'iter_content', 'iter_lines', 'json', 'links', 'next', 'ok', 'raise_for_status', 
    'raw', 'reason', 'request', 'status_code', 'text', 'url'
    '''
    messages = []
    last_message = None
    w = f'https://api.openai.com/v1/conversations/{conversation_id}/items'
    params = {'order': 'asc',
#              'include': 'message.output_text.logprobs',
             'limit': 100}
    while True:
        # Set up params (keep your logic)
        if last_message is not None:
            params["after"] = last_message["id"]
        elif "after" in params:
            del params["after"]
        ## fetch ##
        conv = requests.get(w, headers=headers,params=params)
        conv.raise_for_status()

        text = conv.text
        data = json.loads(text).get('data', [])
        new_messages = list(filter(lambda x: 'content' in x, data))

        # Append new messages
        messages += new_messages

        # Break if no more messages
        if len(new_messages) < params["limit"]:
            break

        # Get last message
        
        if new_messages:
            last_message = messages[-1]
    assert all(len(message["content"]) == 1 for message in messages)

    return messages



def get_messages(client: OpenAI, conversation_id: str) -> list[dict]:
    """Gets messages from a thread.

    :param client: The OpenAI client.
    :param thread_id: The ID of the thread to get messages from.
    :return: A list of messages.
    """
    # Set up
    
    messages = []
    last_message = None
    params = {
        "metadata": {'conversation_id': conversation_id},
    }

    # Get all messages from the thread page by page
    while True:
        # Set up params
        if last_message is not None:
            params["after"] = last_message["id"]
        elif "after" in params:
            del params["after"]

        # Get messages
        new_messages = [
            message.to_dict() for message in client.responses.create(**params)
        ]

        # Append new messages
        messages += new_messages

        # Break if no more messages
        if len(new_messages) < params["limit"]:
            break

        # Get last message
        last_message = messages[-1]
    ## "content": [
#   {"type": "input_text", "text": "What’s the weather?"},
#   {"type": "output_text", "text": "It’s sunny in Boston."}
# ]
    # Verify all message content is length 1
    assert all(len(message["content"]) == 1 for message in messages)
        
    return messages


async def async_get_messages(client: AsyncOpenAI, conversation_id: str) -> list[dict]:
    """Gets messages from a thread.

    :param client: The async OpenAI client.
    :param thread_id: The ID of the thread to get messages from.
    :return: A list of messages.
    """
    # Set up
    messages = []
    last_message = None
    params = {
        "metadata": {'conversation_id': conversation_id},
        "limit": 100,
        
        "order": "asc",
        
    }

    # Get all messages from the thread page by page
    while True:
        # Set up params
        if last_message is not None:
            params["after"] = last_message["id"]
        elif "after" in params:
            del params["after"]

        # Get messages
        new_messages = [
            message.to_dict()
            async for message in client.responses.create(**params)
        ]

        # Append new messages
        messages += new_messages

        # Break if no more messages
        if len(new_messages) < params["limit"]:
            break

        # Get last message
        last_message = messages[-1]

    # Verify all message content is length 1
    assert all(len(message["content"]) == 1 for message in messages)

    return messages


def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string.

    :param string: The text string to count tokens in.
    :param encoding_name: The name of the encoding to use.
    :return: The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens


def update_token_counts(
    token_counts: dict[str, int],
    discussion: list[dict[str, str]],
    response: str,
) -> None:
    """Updates the token counts (in place) with a discussion and response.

    :param token_counts: The token counts to update.
    :param discussion: The discussion to update the token counts with.
    :param response: The response to update the token counts with.
    """
    new_input_token_count = sum(count_tokens(turn["message"]) for turn in discussion)
    new_output_token_count = count_tokens(response)

    token_counts["input"] += new_input_token_count
    token_counts["output"] += new_output_token_count

    token_counts["max"] = max(
        token_counts["max"], new_input_token_count + new_output_token_count
    )


def count_discussion_tokens(
    discussion: list[dict[str, str]],
) -> dict[str, int]:
    """Counts the number of tokens in a discussion.

    :param discussion: The discussion to count tokens in.
    :return: A dictionary of token counts.
    """
    token_counts = {
        "input": 0,
        "output": 0,
        "max": 0,
    }

    for index, turn in enumerate(discussion):
        if turn["agent"] != "User":
            update_token_counts(
                token_counts=token_counts,
                discussion=discussion[:index],
                response=turn["message"],
            )

    return token_counts


def compute_token_cost(
    model: str, input_token_count: int, output_token_count: int
) -> float:
    """Computes the token cost of a model given input and output token counts.

    :param model: The name of the model.
    :param input_token_count: The number of tokens in the input.
    :param output_token_count: The number of tokens in the output.
    :return: The token cost of the model.
    """
    if (
        model not in MODEL_TO_INPUT_PRICE_PER_TOKEN
        or model not in MODEL_TO_OUTPUT_PRICE_PER_TOKEN
    ):
        raise ValueError(f'Cost of model "{model}" not known')

    return (
        input_token_count * MODEL_TO_INPUT_PRICE_PER_TOKEN[model]
        + output_token_count * MODEL_TO_OUTPUT_PRICE_PER_TOKEN[model]
    )


def print_cost_and_time(
    token_counts: dict[str, int],
    model: str,
    elapsed_time: float,
) -> None:
    # Print token counts
    print(f"Input token count: {token_counts['input']:,}")
    print(f"Output token count: {token_counts['output']:,}")
    print(f"Tool token count: {token_counts['tool']:,}")
    print(f"Max token length: {token_counts['max']:,}")

    # Compute and print cost
    try:
        cost = compute_token_cost(
            model=model,
            input_token_count=token_counts["input"] + token_counts["tool"],
            output_token_count=token_counts["output"],
        )
        print(f"Cost: ${cost:.2f}")
    except ValueError as e:
        print(f"Warning: {e}")

    # Print time
    print(f"Time: {int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}")


def compute_finetuning_cost(
    model: str, token_count: int, num_epochs: int = DEFAULT_FINETUNING_EPOCHS
) -> float:
    """Computes the cost of fine-tuning a model.

    :param model: The model that will be finetuned.
    :param token_count: The number of training tokens for finetuning.
    :param num_epochs: Number of finetuning epochs.
    :return: The cost of finetuning.
    """
    if model not in FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN:
        raise ValueError(f'Cost of model "{model}" not known')

    return (
        token_count * FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN[model] * num_epochs
    )

## changed
def convert_messages_to_discussion(
    messages: list[dict],
    assistant_id_to_title: dict[str, str],
) -> list[dict[str, str]]:
    """
    Convert API conversation items (or similar) into your discussion format:
    [{"agent": "<name>", "message": "<text>"}].
    Tries multiple schemas safely.
    """
    def extract_text(msg: dict) -> str:
        """
        Works with Conversations/Responses items where content is a list of parts.
        Collects text from input/output parts and joins with newlines.
        Handles both string `text` and dict `text={"value": ...}`.
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
        except Exception:
            pass

        # If content itself is a string
        if isinstance(content, str):
            return content

        return ""

    def resolve_agent(msg: dict) -> str:
        # Prefer explicit assistant id if present and mapped
        a_id = msg.get("assistant_id")
        if a_id and a_id in assistant_id_to_title:
            return assistant_id_to_title[a_id]

        # Some Responses items carry response_id you mapped earlier
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
    def _text(m: dict) -> str:
        return extract_text(m)

    def _agent(m: dict) -> str:
        return resolve_agent(m)

    return [{"agent": _agent(m), "message": _text(m)} for m in messages]

#     out = []
#     for m in messages:
#         # Only transform actual messages if you're passing raw items
#         if m.get("type") and m["type"] != "message":
#             continue
#         text = extract_text(m)
#         if not text:
#             continue
#         out.append({"agent": resolve_agent(m), "message": text})
    return out

def get_summary(discussion: list[dict[str, str]]) -> str:
    """Get the summary from a discussion.

    :param discussion: The discussion to extract the summary from.
    :return: The summary.
    """
    return discussion[-1]["message"]


def load_summaries(discussion_paths: list[Path]) -> tuple[str, ...]:
    """Load summaries from a list of discussion paths.

    :param discussion_paths: The paths to the discussion JSON files. The summary is the last entry in the discussion.
    :return: A tuple of summaries.
    """
    summaries = []
    for discussion_path in discussion_paths:
        with open(discussion_path, "r") as file:
            discussion = json.load(file)
        summaries.append(get_summary(discussion))
    return tuple(summaries)


def save_meeting(
    save_dir: Path, save_name: str, discussion: list[dict[str, str]]
) -> None:
    """Save a meeting discussion to JSON and Markdown files.

    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param discussion: The discussion to save.
    """
    # Create the save directory if it does not exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the discussion as JSON
    with open(save_dir / f"{save_name}.json", "w") as f:
        json.dump(discussion, f, indent=4)

    # Save the discussion as Markdown
    with open(save_dir / f"{save_name}.md", "w") as file:
        for turn in discussion:
            file.write(f"## {turn['agent']}\n\n{turn['message']}\n\n")

def get_recent_markdown(path: Path):
    md_file = sorted(list(path.glob('*.md')))[-1] # 1, 2,3
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def write_final_summary(save_dir: Path, summary: str, grant_name: str, out_suffix: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f'{grant_name}_{out_suffix}.txt', 'w') as f:
        print(summary, file=f)
    return

def clear_dir(save_dir: Path):
    if save_dir.exists():
        shutil.rmtree(save_dir)
    return
