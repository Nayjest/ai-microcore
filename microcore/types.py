from typing import Callable, Any, Awaitable, Union
from os import PathLike

TplFunctionType = Callable[[Union[PathLike[str], str], Any], str]
"""Function type for rendering prompt templates"""
LLMFunctionType = Callable[[str, Any], str]
"""Function type for requesting LLM synchronously"""
LLMAsyncFunctionType = Callable[[str, Any], Awaitable[str]]
"""Function type for requesting LLM asynchronously"""


class BadAIAnswer(ValueError):
    """Unprocessable response generated by the LLM"""

    def __init__(self, message: str = None, details: str = None):
        self.message = str(message or "Unprocessable response generated by the LLM")
        self.details = details
        super().__init__(self.message + (f": {self.details}" if self.details else ""))

    def __str__(self):
        return self.message + (f": {self.details}" if self.details else "")


class BadAIJsonAnswer(BadAIAnswer):
    def __init__(
        self, message: str = "Invalid JSON generated by the LLM", details=None
    ):
        super().__init__(message, details)
