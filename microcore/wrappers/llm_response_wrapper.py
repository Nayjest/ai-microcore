from typing import Any

from ..types import BadAIAnswer, TPrompt
from ..json_parsing import parse_json
from ..utils import ExtendedString, ConvertableToMessage, extract_number
from ..message_types import Role, AssistantMsg


class DictFromLLMResponse(dict):
    llm_response: "LLMResponse"

    def from_llm_response(self, llm_response: "LLMResponse"):
        self.llm_response = llm_response
        return self


class LLMResponse(ExtendedString, ConvertableToMessage):
    """
    Response from the Large Language Model.

    If treated as a string, it returns the text generated by the LLM.

    Also, it contains all fields returned by the API accessible as an attributes.

    See fields returned by the OpenAI:

    - https://platform.openai.com/docs/api-reference/completions/object
    - https://platform.openai.com/docs/api-reference/chat/object
    """

    role: Role
    content: str
    prompt: TPrompt
    gen_duration: float

    def __new__(cls, string: str, attrs: dict = None):
        attrs = {
            **(attrs or {}),
            "role": Role.ASSISTANT,
            "content": str(string),
            "prompt": None,
            # generation duration in seconds (float), used in metrics
            "gen_duration": None,
        }
        obj = ExtendedString.__new__(cls, string, attrs)
        return obj

    def parse_json(
        self, raise_errors: bool = True, required_fields: list[str] = None
    ) -> list | dict | float | int | str | DictFromLLMResponse:
        res = parse_json(self.content, raise_errors, required_fields)
        if isinstance(res, dict):
            res = DictFromLLMResponse(res)
            res.llm_response = self
        return res

    def parse_number(
        self,
        default=BadAIAnswer,
        position="last",
        dtype: type | str = float,
        rounding: bool = False,
    ) -> int | float | Any:
        return extract_number(self.content, default, position, dtype, rounding)

    def as_message(self) -> AssistantMsg:
        return self.as_assistant
