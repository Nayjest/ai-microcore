import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import tiktoken

from ..utils import ExtendedString


INT32_MAX = 2**31 - 1  # 2147483647


class SearchResults(list):
    def fit_to_token_size(
        self,
        max_tokens: int,
        min_documents: int = None,
        for_model: str = None,
        encoding: str | tiktoken.Encoding = None,
        verbose=True,
    ):
        from ..tokenizing import fit_to_token_size

        records, removed = fit_to_token_size(
            self,
            max_tokens=max_tokens,
            min_documents=min_documents,
            for_model=for_model,
            encoding=encoding,
        )
        if verbose and removed:
            logging.info(
                "For fitting %d records to %d tokens, %d records was removed",
                len(self),
                max_tokens,
                removed,
            )
        return SearchResults(list(records))


class SearchResult(ExtendedString):
    """
    String containing the search result with additional information in attributes

    Attributes:
        id (str): document (text) identifier in embedding database
        distance (float): The distance between the query and the search result
        metadata (dict): A dictionary containing document metadata
    """

    id: str
    distance: float
    metadata: dict


@dataclass
class AbstractEmbeddingDB(ABC):
    """
    Base class for embedding databases
    """

    @abstractmethod
    def search(
        self,
        collection: str,
        query: str | list,
        n_results: int = 5,
        where: dict = None,
        **kwargs,
    ) -> list[str | SearchResult]:
        """
        Similarity search

        Args:
            collection (str): collection name
            query (str | list): query string or list of query strings
            n_results (int): number of results to return
            where (dict): filter results by metadata
            **kwargs: additional arguments
        """

    def find(self, *args, **kwargs) -> SearchResults | list[str | SearchResult]:
        """
        Alias for `search`
        """
        return self.search(*args, **kwargs)

    @abstractmethod
    def get(
        self,
        collection: str,
        ids: list[str] | str = None,
        limit: int = None,
        offset: int = None,
        where: dict = None,
        **kwargs,
    ) -> list[str | SearchResult] | str | SearchResult | None:
        """
        Get documents

        Args:
            collection (str): collection name
            ids (list[str] | str): document id or list of document ids
            limit (int): maximum number of documents to return
            offset (int): number of documents to skip
            where (dict): filter results by metadata

        Returns:
            List of documents or single document
        """

    def find_all(
        self,
        collection: str,
        query: str | list,
        where: dict = None,
        **kwargs,
    ) -> SearchResults | list[str | SearchResult]:
        return self.search(
            collection, query, n_results=INT32_MAX, where=where, **kwargs
        )

    @abstractmethod
    def get_all(self, collection: str) -> SearchResults | list[str | SearchResult]:
        """Return all documents in the collection"""

    def save(self, collection: str, text: str, metadata: dict = None):
        """Save a single document in the collection"""
        self.save_many(collection, [(text, metadata)])

    @abstractmethod
    def save_many(self, collection: str, items: list[tuple[str, dict] | str]):
        """Save multiple documents in the collection"""

    @abstractmethod
    def clear(self, collection: str):
        """Clear the collection"""

    def find_one(self, collection: str, query: str | list) -> str | SearchResult | None:
        """
        Find most similar document in the collection

        Returns:
            Most similar document or None if collection is empty
        """
        return next(iter(self.search(collection, query, 1)), None)

    @abstractmethod
    def count(self, collection: str) -> int:
        """
        Count the number of documents in the collection

        Returns:
            Number of documents in the collection, 0 if collection does not exist
        """

    @abstractmethod
    def delete(self, collection: str, what: str | list[str] | dict):
        """
        Delete documents from the collection

        Args:
            collection (str): collection name
            what (str | list[str] | dict): id, list ids or metadata query
        """

    @abstractmethod
    def collection_exists(self, collection: str) -> bool:
        """
        Check if the collection exists

        Returns:
            True if collection exists, False otherwise
        """

    def has_content(self, collection: str) -> bool:
        """
        Check if the collection exists anf contains any documents

        Returns:
            True if collection exists and contains documents, False otherwise
        """
        return self.count(collection) > 0
