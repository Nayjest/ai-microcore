"""
Message classes for OpenAI-like Chat APIs.
"""
import abc
from enum import Enum
from dataclasses import dataclass, field
from typing import ClassVar, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class Role(str, Enum):
    """
    Enum representing the possible roles in a chat message.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self):
        return self.value


DEFAULT_MESSAGE_ROLE = Role.USER
"""Default role for messages if not specified (USER)."""


class _PydanticPassthrough:
    """
    Mixin that allows Pydantic to accept instances of this class without transformation.
    """
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler  # pylint: disable=unused-argument
    ) -> core_schema.CoreSchema:
        return core_schema.any_schema()


class MsgContentPart(_PydanticPassthrough):
    """
    Base class for individual parts of multipart message content.
    """


class MsgContent(_PydanticPassthrough):
    """
    Base class for message content.
    """


class MsgMultipartContent(MsgContent, abc.ABC):
    """
    Abstract base class for multipart message content.
    """
    @abc.abstractmethod
    def parts(self) -> list[MsgContentPart]:
        """
        Returns the list of content parts.
        """
        raise NotImplementedError()


TMsgContentPart = str | dict | MsgContentPart
"""Type alias for a single content part: string, dictionary, or MsgContentPart."""

TMsgContent = str | MsgContent | list[TMsgContentPart]
"""Type alias for message content: string, MsgContent, or list of content parts."""


@dataclass
class Msg(_PydanticPassthrough):
    """
    Represents a message in a chat conversation.
    """
    role: str = field(default=DEFAULT_MESSAGE_ROLE)
    content: TMsgContent = field(default="")

    DICT_FACTORY: ClassVar = dict

    def __str__(self):
        return str(self.content)

    def strip(self):
        """Strips whitespace from the content if it is a string."""
        if isinstance(self.content, str):
            self.content = self.content.strip()
        return self


@dataclass
class SysMsg(Msg):
    """System message."""
    role: str = field(default=Role.SYSTEM, init=False)


@dataclass
class UserMsg(Msg):
    """User message."""
    role: str = field(default=Role.USER, init=False)


@dataclass
class AssistantMsg(Msg):
    """Assistant message."""
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
