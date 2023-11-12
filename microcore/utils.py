import builtins
import dataclasses
import inspect
import json
import re


def is_chat_model(model: str) -> bool:
    """Detects if model is chat model or text completion model"""
    completion_keywords = ["instruct", "davinci", "babbage", "curie", "ada"]
    return not any(keyword in model.lower() for keyword in completion_keywords)


class ExtendedString(str):
    """
    Provides a way of extending string with attributes and methods
    """

    def __new__(cls, string: str, attrs: dict = None):
        """
        Allows string to have attributes.
        """
        obj = str.__new__(cls, string)
        if attrs:
            for k, v in attrs.items():
                setattr(obj, k, v)
        return obj

    def __getattr__(self, item):
        """
        Provides chaining of global functions
        """
        global_func = inspect.currentframe().f_back.f_globals.get(item) or vars(
            builtins
        ).get(item, None)
        if callable(global_func):

            def method_handler(*args, **kwargs):
                res = global_func(self, *args, **kwargs)
                if isinstance(res, str) and not isinstance(res, ExtendedString):
                    res = ExtendedString(res)
                return res

            return method_handler

        # If there's not a global function with that name, raise an AttributeError as usual
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )


class DataclassEncoder(json.JSONEncoder):
    """@private"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


json.JSONEncoder.default = DataclassEncoder().default


def parse(text: str, field_format: str = r"\[\[(.*?)\]\]") -> dict:
    """
    Parse a document divided into sections and convert it into a dictionary.
    """
    pattern = rf"{field_format}\n(.*?)(?=\n{field_format}|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return {key.strip().lower(): value for key, value, _ in matches}
