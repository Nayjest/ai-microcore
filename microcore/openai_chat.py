""" Message classes for OpenAI chat API if you prefer an  object-oriented approach """
import dataclasses
import json
import microcore.prepare_llm_args


class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


json.JSONEncoder.default = DataclassEncoder().default


@dataclasses.dataclass
class Msg:
    role: str = microcore.prepare_llm_args.default_chat_message_role
    content: str = ''
    def __str__(self): return str(self.content)


class _BaseMsg(Msg):
    def __init__(self, content: str): self.content = content


class SysMsg(_BaseMsg):
    role: str = 'system'


class UserMsg(_BaseMsg):
    role: str = 'user'


class AssistantMag(_BaseMsg):
    role: str = 'assistant'

