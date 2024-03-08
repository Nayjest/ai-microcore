""" Message classes for OpenAI Chat API """

from dataclasses import dataclass, field


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


DEFAULT_MESSAGE_ROLE = Role.USER


@dataclass
class Msg:
    role: str = field(default=DEFAULT_MESSAGE_ROLE)
    content: str = field(default="")

    def __str__(self):
        return str(self.content)


@dataclass
class SysMsg(Msg):
    role: str = field(default=Role.SYSTEM, init=False)


@dataclass
class UserMsg(Msg):
    role: str = field(default=Role.USER, init=False)


@dataclass
class AssistantMsg(Msg):
    role: str = field(default=Role.ASSISTANT, init=False)
