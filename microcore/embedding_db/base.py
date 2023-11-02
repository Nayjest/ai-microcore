from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..extended_string import ExtendedString


class SearchResult(ExtendedString):
    id: str
    distance: float
    metadata: dict


@dataclass
class EmbeddingDB(ABC):
    @abstractmethod
    def search(
        self,
        collection: str,
        query: str | list,
        n_results: int = 5,
        where: dict = None,
        **kwargs,
    ) -> list[str | SearchResult]:
        pass

    @abstractmethod
    def get_all(self, collection: str) -> list[str | SearchResult]:
        pass

    def save(self, collection: str, text: str, metadata: dict = None):
        self.save_many(collection, [(text, metadata)])

    @abstractmethod
    def save_many(self, collection: str, items: list[tuple[str, dict] | str]):
        pass

    @abstractmethod
    def clear(self, collection: str):
        pass

    def find_one(self, collection: str, query: str | list) -> str | None:
        return next(iter(self.search(collection, query, 1)), None)
