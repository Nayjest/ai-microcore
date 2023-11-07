from .message_types import DEFAULT_MESSAGE_ROLE


def prepare_prompt(prompt) -> str:
    """Converts prompt to string for LLM completion API"""
    return "\n".join(
        [
            str(p["content"]) if isinstance(p, dict) and "content" in p else str(p)
            for p in (prompt if isinstance(prompt, list) else [prompt])
        ]
    )


def prepare_chat_messages(prompt):
    """Converts prompt to messages for LLM chat API (OpenAI)"""
    return [
        dict(role=DEFAULT_MESSAGE_ROLE, content=msg) if isinstance(msg, str) else msg
        for msg in (prompt if isinstance(prompt, list) else [prompt])
    ]
