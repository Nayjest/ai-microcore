"""
# Minimalistic Foundation for AI Applications

**microcore** is a collection of python adapters for Large Language Models
and Semantic Search APIs allowing to
communicate with these services convenient way, make it easily switchable
and separate business logic from implementation details.
"""
import os

from .embedding_db import SearchResult, AbstractEmbeddingDB
from .file_storage import storage
from ._env import configure, env
from .logging import use_logging
from .message_types import UserMsg, AssistantMsg, SysMsg, Msg
from .config import ApiType, LLMConfigError
from .types import BadAIJsonAnswer, BadAIAnswer
from .wrappers.prompt_wrapper import PromptWrapper
from .wrappers.llm_response_wrapper import LLMResponse
from ._llm_functions import llm, allm


def tpl(file: os.PathLike[str] | str, **kwargs) -> str | PromptWrapper:
    """Renders a prompt template using the provided parameters."""
    return PromptWrapper(env().tpl_function(file, **kwargs))


def use_model(name: str):
    """Switches language model"""
    env().config.MODEL = name
    env().config.LLM_DEFAULT_ARGS["model"] = name


def validate_config():
    """
    Validates current MicroCore configuration

    Raises:
        `LLMConfigError` if configuration is invalid
    """
    env().config.validate()


class _EmbeddingDBProxy(AbstractEmbeddingDB):
    def get_all(self, collection: str) -> list[str | SearchResult]:
        return env().texts.get_all(collection)

    def search(
        self,
        collection: str,
        query: str | list,
        n_results: int = 5,
        where: dict = None,
        **kwargs,
    ) -> list[str | SearchResult]:
        return env().texts.search(collection, query, n_results, where, **kwargs)

    def save_many(self, collection: str, items: list[tuple[str, dict] | str]):
        return env().texts.save_many(collection, items)

    def save(self, collection: str, text: str, metadata: dict = None):
        return env().texts.save(collection, text, metadata)

    def clear(self, collection: str):
        return env().texts.clear(collection)


texts = _EmbeddingDBProxy()
"""Embedding database, see `microcore.embedding_db.AbstractEmbeddingDB`"""

__all__ = [
    "llm",
    "allm",
    "tpl",
    "texts",
    "configure",
    "validate_config",
    "storage",
    "use_model",
    "use_logging",
    "env",
    "Msg",
    "UserMsg",
    "SysMsg",
    "AssistantMsg",
    "ApiType",
    "BadAIJsonAnswer",
    "BadAIAnswer",
    "LLMConfigError",
    "LLMResponse",
    "PromptWrapper",
    # submodules
    "embedding_db",
    "file_storage",
    "message_types",
    "utils",
    "config",
    "types",
    # "wrappers",
]

__version__ = "0.7.0"
