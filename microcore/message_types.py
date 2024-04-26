""" Message classes for OpenAI Chat API """

from dataclasses import dataclass, field


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


DEFAULT_MESSAGE_ROLE = Role.USER


@dataclass
class Msg:
    dict_factory = dict
    role: str = field(default=DEFAULT_MESSAGE_ROLE)
    content: str = field(default="")

    def __str__(self):
        return str(self.content)

    def strip(self):
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

    dict_factory = _PartialMsgDict
    placeholder = "<|placeholder|>"
    variants_splitter = "<|or|>"

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
