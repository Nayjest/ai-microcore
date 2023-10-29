"""Minimalistic core for large language model applications"""
import os

from .embedding_db.base import EmbeddingDB, SearchResult
from .storage import storage
from .env import env, configure
from .logging import use_logging
from .storage import storage # noqa


def llm(prompt, **kwargs) -> str:
    [h(prompt, **kwargs) for h in env().llm_before_handlers]
    response = env().llm_function(prompt, **kwargs)
    [h(response) for h in env().llm_after_handlers]
    return response


def tpl(file: os.PathLike[str] | str, **kwargs) -> str: return env().tpl_function(file, **kwargs)


def ssearch(collection: str, query: str | list, n: int = 5) -> list[str]:
    return env().embeddings.search(collection, query, n)


def use_model(name: str):
    env().config.MODEL = name
    env().config.LLM_DEFAULT_ARGS['model'] = name


class _EmbeddingProxy(EmbeddingDB):

    def search(
            self,
            collection: str,
            query: str | list,
            n_results: int = 5,
            where: dict = None,
            **kwargs
    ) -> list[str | SearchResult]:
        return env().embeddings.search(collection, query, n_results, where, **kwargs)

    def save_many(self, collection: str, items: list[tuple[str, dict] | str]):
        return env().embeddings.save_many(collection, items)

    def clean(self, collection: str):
        return env().embeddings.clean(collection)


embeddings = _EmbeddingProxy()


__all__ = ['configure', 'llm', 'tpl', 'storage', 'use_model', 'use_logging', 'env', 'embeddings']
