"""
# Minimalistic Foundation for AI Applications

**microcore** is a collection of python adapters for Large Language Models
and Semantic Search APIs allowing to
communicate with these services convenient way, make it easily switchable
and separate business logic from implementation details.
"""

import os
import microcore.ui  # noqa
from .embedding_db import SearchResult, AbstractEmbeddingDB
from .file_storage import storage
from ._env import configure, env, config
from .logging import use_logging
from .message_types import UserMsg, AssistantMsg, SysMsg, Msg
from .configuration import ApiType, LLMConfigError, Config
from .types import BadAIJsonAnswer, BadAIAnswer
from .wrappers.prompt_wrapper import PromptWrapper
from .wrappers.llm_response_wrapper import LLMResponse
from ._llm_functions import llm, allm, llm_parallel
from .utils import parse, dedent


def tpl(file: os.PathLike[str] | str, **kwargs) -> str | PromptWrapper:
    """Renders a prompt template using the provided parameters."""
    return PromptWrapper(env().tpl_function(file, **kwargs))


def prompt(template_str: str, remove_indent=True, **kwargs) -> str | PromptWrapper:
    """Renders a prompt template from string using the provided parameters."""
    if remove_indent:
        template_str = dedent(template_str)
    return PromptWrapper(env().jinja_env.from_string(template_str).render(**kwargs))


fmt = prompt


def use_model(name: str):
    """Switches language model"""
    config().MODEL = name
    config().LLM_DEFAULT_ARGS["model"] = name


def validate_config():
    """
    Validates current MicroCore configuration

    Raises:
        `LLMConfigError` if configuration is invalid
    """
    config().validate()


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

    def find(self, *args, **kwargs) -> list[str | SearchResult]:
        return self.search(*args, **kwargs)

    def find_all(
        self,
        collection: str,
        query: str | list,
        where: dict = None,
        **kwargs,
    ) -> list[str | SearchResult]:
        return env().texts.find_all(collection, query, where, **kwargs)

    def save_many(self, collection: str, items: list[tuple[str, dict] | str]):
        return env().texts.save_many(collection, items)

    def save(self, collection: str, text: str, metadata: dict = None):
        return env().texts.save(collection, text, metadata)

    def clear(self, collection: str):
        return env().texts.clear(collection)

    def count(self, collection: str) -> int:
        return env().texts.count(collection)

    def delete(self, collection: str, what: str | list[str] | dict):
        return env().texts.delete(collection, what)


texts = _EmbeddingDBProxy()
"""Embedding database, see `microcore.embedding_db.AbstractEmbeddingDB`"""

__all__ = [
    "llm",
    "allm",
    "llm_parallel",
    "tpl",
    "prompt",
    "fmt",
    "texts",
    "configure",
    "validate_config",
    "storage",
    "use_model",
    "use_logging",
    "env",
    "config",
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
    "parse",
    "dedent",
    # submodules
    "embedding_db",
    "file_storage",
    "message_types",
    "utils",
    "configuration",
    "Config",
    "types",
    "ui",
    # "wrappers",
]

__version__ = "3.5.0"
