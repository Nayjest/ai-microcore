from dataclasses import asdict

from .message_types import DEFAULT_MESSAGE_ROLE, Msg


def prepare_prompt(prompt) -> str:
    """Converts prompt to string for LLM completion API"""
    return "\n".join(
        [
            str(p["content"]) if isinstance(p, dict) and "content" in p else str(p)
            for p in (prompt if isinstance(prompt, list) else [prompt])
        ]
    )


def prepare_chat_messages(prompt) -> list[dict]:
    """Converts prompt to messages for LLM chat API (OpenAI)"""
    messages = prompt if isinstance(prompt, list) else [prompt]
    return [
        (
            dict(role=DEFAULT_MESSAGE_ROLE, content=msg)
            if isinstance(msg, str)
            else (
                asdict(msg, dict_factory=msg.dict_factory)
                if isinstance(msg, Msg)
                else msg
            )
        )
        for msg in messages
    ]
