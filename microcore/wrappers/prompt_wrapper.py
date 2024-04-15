from .._llm_functions import allm, llm
from ..utils import ExtendedString, ConvertableToMessage


class PromptWrapper(ExtendedString, ConvertableToMessage):
    def to_llm(self, **kwargs):
        """
        Send prompt to Large Language Model, see `llm`
        """
        return llm(self, **kwargs)

    async def to_allm(self, **kwargs):
        """
        Send prompt to Large Language Model asynchronously, see `allm`
        """
        return await allm(self, **kwargs)
