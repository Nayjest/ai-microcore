""" Message classes for OpenAI Chat API """
import dataclasses


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


default_chat_message_role = Role.USER


@dataclasses.dataclass
class Msg:
    role: str = default_chat_message_role
    content: str = ""

    def __str__(self):
        return str(self.content)


class _BaseMsg(Msg):
    def __init__(self, content: str):
        super().__init__()
        self.content = content


class SysMsg(_BaseMsg):
    role: str = Role.SYSTEM


class UserMsg(_BaseMsg):
    role: str = Role.USER


class AssistantMsg(_BaseMsg):
    role: str = Role.ASSISTANT


__all__ = [
    "Msg",
    "UserMsg",
    "SysMsg",
    "AssistantMsg",
    "Role",
    "default_chat_message_role",
]
