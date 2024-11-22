from typing import TYPE_CHECKING, Union

from .._llm_functions import allm, llm
from ..utils import ExtendedString, ConvertableToMessage
if TYPE_CHECKING:
    from .wrappers.llm_response_wrapper import LLMResponse  # noqa: F401


class PromptWrapper(ExtendedString, ConvertableToMessage):
    """
    A utility class that wraps a prompt string, extending it with convenient methods
    for enhanced functionality.
    """
    def to_llm(self, **kwargs) -> Union[str, "LLMResponse"]:
        """
        Send prompt to Large Language Model, see `llm`
        """
        return llm(self, **kwargs)

    async def to_allm(self, **kwargs) -> Union[str, "LLMResponse"]:
        """
        Send prompt to Large Language Model asynchronously, see `allm`
        """
        return await allm(self, **kwargs)
