"""Base classes for LLM inference API clients."""
import abc
from dataclasses import dataclass
from typing import Any

from .configuration import Config
from .types import TPrompt
from .message_types import TMsgContent, TMsgContentPart, MsgMultipartContent
from .wrappers.llm_response_wrapper import LLMResponse
from ._prepare_llm_args import prompt_to_message_dicts


class BaseAsyncAIClient(abc.ABC):
    """
    Base class for asynchronous LLM inference API clients.

    Async client instance is available via the `aio` attribute of the synchronous client.
    """

    async def __call__(self, prompt: TPrompt, **kwargs) -> LLMResponse:
        """
        Using client instance as a callable calls the generate method.
        """
        return await self.generate(prompt, **kwargs)

    @abc.abstractmethod
    async def generate(self, prompt: TPrompt, **kwargs) -> LLMResponse:
        raise NotImplementedError()

    async def model_names(self) -> list[str]:
        return list((await self.load_models()).keys())

    @abc.abstractmethod
    async def load_models(self) -> dict:
        raise NotImplementedError()


@dataclass
class BaseAIClient(abc.ABC):
    """
    Base class for LLM inference API clients.
    """
    aio: BaseAsyncAIClient
    """Async client instance."""
    config: Config

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, prompt: TPrompt, **kwargs) -> LLMResponse:
        """
        Using client instance as a callable calls the generate method.
        """
        return self.generate(prompt, **kwargs)

    @abc.abstractmethod
    def generate(self, prompt: TPrompt, **kwargs) -> LLMResponse:
        raise NotImplementedError()

    def model_names(self) -> list[str]:
        """
        Get a list of available model names from the LLM inference API.
        """
        return list(self.load_models().keys())

    @abc.abstractmethod
    def load_models(self) -> dict[str, Any]:
        """
        Load available models from the LLM inference API.
        Returns a dictionary mapping model IDs to their details.
        """
        raise NotImplementedError()


class BaseAIChatClient(BaseAIClient, abc.ABC):

    def convert_prompt_to_chat_input(self, prompt: TPrompt) -> list[dict | Any]:
        messages = prompt_to_message_dicts(prompt, strict=False)
        return [self._convert_message(m) for m in messages]

    def _convert_message(self, message: dict) -> dict | Any:
        message["content"] = self._convert_message_content(message["content"])
        return message

    def _convert_message_content_part(
        self,
        content_part: TMsgContentPart,
        converted_content: list = None,  # pylint: disable=unused-argument
    ) -> Any | list[Any] | None:
        """
        Convert the content block into a format suitable for the LLM inference chat API.
        """
        return content_part

    def _convert_message_content(self, message_content: TMsgContent) -> Any:
        """
        Convert message content to format suitable for the LLM inference chat API.
        """
        # Split multipart content into parts
        if isinstance(message_content, MsgMultipartContent):
            message_content = message_content.parts()
        if not isinstance(message_content, list):
            message_content = [message_content]

        converted_content = []
        for part in message_content:
            part = self._convert_message_content_part(part, converted_content)
            if part:
                if isinstance(part, list):
                    converted_content += part
                else:
                    converted_content.append(part)
        return converted_content
