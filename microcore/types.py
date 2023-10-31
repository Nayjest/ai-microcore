from typing import Callable, Any, Awaitable
from os import PathLike
from typing import Union

"""Definition of tpl function used to render templates"""
TplFunctionType = Callable[[Union[PathLike[str], str], Any], str]
LLMFunctionType = Callable[[str, Any], str]
LLMAsyncFunctionType = Callable[[str, Any], Awaitable[str]]