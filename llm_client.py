"""
llm_client.py

Talks to the LLM — any OpenAI-compatible chat completions endpoint,
picked entirely by the three LLM_* settings in .env (see config.py).
Swapping providers (DeepSeek, Groq, OpenAI, etc.) is a .env change, not
a code change, because they all speak the same API shape — that's why,
unlike STT/TTS, there's no separate provider file or dispatch here.

This file turns the SDK's own response objects into two plain,
dependency-free dataclasses (ToolCall and LLMReply) before handing
anything back to the rest of the app. Nowhere else in this codebase
should need to import anything from the `openai` package or know what
its response objects look like — if the LLM SDK ever changes shape,
this is the only file that should need to change with it.
"""

import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

import config
from retry import retry_once


@dataclass
class ToolCall:
    """
    One tool the LLM wants to run this turn.

    Fields:
        id (str): a unique identifier the API gave this specific call.
        Needed later to send that tool's result back to the right place
        in the conversation — the API matches results to calls by this
        id, not by order or by name.
        name (str): which tool to run, e.g. "propose_action".
        arguments (dict): the arguments the LLM filled in for that tool,
        already parsed from JSON text into a normal Python dict.
    """
    id: str
    name: str
    arguments: dict


@dataclass
class LLMReply:
    """
    What the LLM sent back for one turn.

    Fields:
        text (str | None): what the LLM wants to say out loud, if
        anything. Can be None if it only wants to call a tool and hasn't
        said anything yet.
        tool_calls (list[ToolCall]): tools the LLM wants to run this
        turn. Empty list if it just replied with text and didn't ask to
        use a tool.
        raw_message (dict): the assistant's message exactly as the API
        gave it back, unchanged. When continuing a conversation that
        involved a tool call, append THIS to the message history — not
        something reconstructed from `text`/`tool_calls` — because the
        API needs the exact original shape back to match up tool results
        correctly on the next turn.
    """
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)


# max_retries=0 because this app has its own single shared retry policy
# (see retry.py) that every external call goes through uniformly. The
# SDK's own built-in retry (2 by default) would silently stack on top of
# ours otherwise, making the real number of attempts different for the
# LLM than for every other external call in the app for no good reason.
_client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=0)


async def get_reply(messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
    """
    Send the conversation so far to the LLM and get back what it wants
    to do next — speak, call a tool, or both.

    Takes:
        messages (list[dict]): the conversation so far, in standard
        OpenAI chat format (each entry has a "role" and "content", plus
        "tool_calls"/"tool_call_id" on tool-related turns).
        tools (list[dict] | None): tool definitions the LLM is allowed
        to call this turn, in OpenAI's function-calling format. Leave as
        None when the LLM should only be able to reply with text.

    Returns:
        LLMReply: see above.

    Can this fail: yes. The actual network call goes through
    retry_once() (see retry.py) — most failures here are transient
    network trouble, and retrying once is cheap. If both attempts fail,
    raises RuntimeError with the underlying error included.
    """
    request = {"model": config.LLM_MODEL, "messages": messages}
    if tools:
        request["tools"] = tools

    async def call_the_api():
        return await _client.chat.completions.create(**request)

    response = await retry_once(call_the_api, "Calling the LLM")
    message = response.choices[0].message

    tool_calls = [
        ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=json.loads(call.function.arguments),
        )
        for call in (message.tool_calls or [])
    ]

    return LLMReply(
        text=message.content,
        tool_calls=tool_calls,
        raw_message=message.model_dump(),
    )