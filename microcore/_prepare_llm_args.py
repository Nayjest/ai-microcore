from dataclasses import asdict
from typing import Any

from .message_types import DEFAULT_MESSAGE_ROLE, Msg, MsgContent
from .types import TPrompt


def prepare_prompt(prompt) -> str:
    """Converts prompt to string for LLM completion API"""
    return "\n".join(
        [
            str(p["content"]) if isinstance(p, dict) and "content" in p else str(p)
            for p in (prompt if isinstance(prompt, list) else [prompt])
        ]
    )


def prompt_item_to_message_dict(item: Any, strict=False) -> dict | Any:
    """
    Convert a single prompt item to message dict for LLM inference chat API (OpenAI-like).
    Args:
        item: The prompt item to convert. Can be a string, Msg instance, or dict.
        strict: If True, raises TypeError for unsupported types. If False, returns the item as is.
    Returns:
        A dict representing the message,
        or the original item if not convertible and strict is False.
    """
    if isinstance(item, Msg):
        message_dict = asdict(item, dict_factory=item.DICT_FACTORY)
    elif isinstance(item, dict):
        message_dict = item
    else:
        if strict and not isinstance(item, str | MsgContent):
            raise TypeError(f"Unsupported message type: {type(item)}")
        message_dict = dict(role=DEFAULT_MESSAGE_ROLE, content=item)
    return message_dict


def prompt_to_message_dicts(prompt: TPrompt, strict=False) -> list[dict | Any]:
    """
    Convert prompt to messages for LLM inference chat API (OpenAI-like).
    Args:
        prompt: The prompt to convert. Can be a string, Msg instance, dict, or list of these.
        strict: If True, raises TypeError for unsupported types. If False, returns the item as is.
    """
    message_like_items: list[Any] = prompt if isinstance(prompt, list) else [prompt]
    return [prompt_item_to_message_dict(item, strict=strict) for item in message_like_items]
