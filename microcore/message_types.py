"""Message classes for OpenAI Chat API"""
import abc
from enum import Enum
from dataclasses import dataclass, field
from typing import ClassVar


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self):
        return self.value


DEFAULT_MESSAGE_ROLE = Role.USER


class MsgContentPart:
    pass


class MsgContent:
    pass


class MsgMultipartContent(MsgContent,  abc.ABC):
    @abc.abstractmethod
    def parts(self) -> list[MsgContentPart]:
        raise NotImplementedError()


TMsgContentPart = str | dict | MsgContentPart

TMsgContent = str | MsgContent | list[TMsgContentPart]


@dataclass
class Msg:
    content: TMsgContent = field(default="")
    role: str = field(default=DEFAULT_MESSAGE_ROLE)

    DICT_FACTORY: ClassVar = dict

    def __str__(self):
        return str(self.content)

    def strip(self):
        if isinstance(self.content, str):
            self.content = self.content.strip()
        return self


@dataclass
class SysMsg(Msg):
    role: str = field(default=Role.SYSTEM, init=False)


@dataclass
class UserMsg(Msg):
    role: str = field(default=Role.USER, init=False)


@dataclass
class AssistantMsg(Msg):
    role: str = field(default=Role.ASSISTANT, init=False)


class PartialMsg(AssistantMsg):
    """A message that is not fully formed yet."""

    class _PartialMsgDict(dict):
        is_partial = True
        """Custom dictionary class to handle additional properties"""

    placeholder = "<|placeholder|>"
    variants_splitter = "<|or|>"

    DICT_FACTORY: ClassVar = _PartialMsgDict

    @staticmethod
    def split_prefix_and_suffixes(content: str):
        parts = content.split(PartialMsg.placeholder)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        suffixes = suffix.split(PartialMsg.variants_splitter) if suffix else []
        return prefix, suffixes

    def prefix_and_suffixes(self):
        return self.split_prefix_and_suffixes(self.content)

    def prefix(self):
        prefix, _ = self.prefix_and_suffixes()
        return prefix

    def suffixes(self):
        _, suffixes = self.prefix_and_suffixes()
        return suffixes
