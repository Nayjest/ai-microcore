import json
from typing import Any

from ..types import BadAIJsonAnswer
from ..extended_string import ExtendedString
from ..message_types import Role


class LLMResponse(ExtendedString):
    def __new__(cls, string: str, attrs: dict = None):
        obj = ExtendedString.__new__(cls, string, attrs)
        # Same fields like in Msg
        setattr(obj, "role", Role.ASSISTANT)
        setattr(obj, "content", str(string))
        return obj

    def parse_json(
        self, raise_errors: bool = True, required_fields: list[str] = None
    ) -> list | dict | Any:
        try:
            res = json.loads(str(self.content))
            if required_fields:
                if not isinstance(res, dict):
                    raise BadAIJsonAnswer("Not an object")
                for field in required_fields:
                    if field not in res:
                        raise BadAIJsonAnswer(f'Missing field "{field}"')
            return res
        except json.decoder.JSONDecodeError:
            if raise_errors:
                raise BadAIJsonAnswer()
            return False