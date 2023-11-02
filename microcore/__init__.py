"""Minimalistic core for large language model applications"""
import os

from .embedding_db.base import EmbeddingDB, SearchResult
from .storage import storage
from .env import env, configure
from .logging import use_logging
from .storage import storage  # noqa
from .message_types import UserMsg, AssistantMsg, SysMsg, Msg
from .config import ApiType, LLMConfigError
from .types import BadAIJsonAnswer, BadAIAnswer
from .wrappers.prompt_wrapper import PromptWrapper
from .llm_functions import llm, allm


def tpl(file: os.PathLike[str] | str, **kwargs) -> str | PromptWrapper:
    return PromptWrapper(env().tpl_function(file, **kwargs))


def use_model(name: str):
    env().config.MODEL = name
    env().config.LLM_DEFAULT_ARGS["model"] = name


def validate_config():
    env().config.validate()


class _EmbeddingDBProxy(EmbeddingDB):
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


__all__ = [
    "llm",
    "allm",
    "configure",
    "validate_config",
    "tpl",
    "storage",
    "use_model",
    "use_logging",
    "env",
    "texts",
    "Msg",
    "UserMsg",
    "SysMsg",
    "AssistantMsg",
    "ApiType",
    "BadAIJsonAnswer",
    "BadAIAnswer",
    "LLMConfigError",
]
