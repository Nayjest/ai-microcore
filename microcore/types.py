from typing import Callable, Any, Awaitable, Union
from os import PathLike

"""Definition of tpl function used to render templates"""
TplFunctionType = Callable[[Union[PathLike[str], str], Any], str]
LLMFunctionType = Callable[[str, Any], str]
LLMAsyncFunctionType = Callable[[str, Any], Awaitable[str]]


class BadAIAnswer(ValueError):
    def __init__(self, message: str = None, details: str = None):
        self.message = message or "Unprocessable response generated by the LLM"
        self.details = details
        super().__init__(message)

    def __str__(self):
        return self.message + f": {self.details}" if self.details else ""

    def safe_error_message(self):
        return str(self)


class BadAIJsonAnswer(BadAIAnswer):
    def __init__(
        self, message: str = "Invalid JSON generated by the LLM", details=None
    ):
        super().__init__(message, details)