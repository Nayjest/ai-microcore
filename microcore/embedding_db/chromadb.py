import logging
from dataclasses import dataclass
import uuid

import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions
from ..configuration import Config
from .. import SearchResult, SearchResults, AbstractEmbeddingDB


@dataclass
class ChromaEmbeddingDB(AbstractEmbeddingDB):
    config: Config
    embedding_function: embedding_functions.EmbeddingFunction = None
    client: chromadb.Client = None

    def __post_init__(self):
        if self.config.EMBEDDING_DB_HOST:
            logging.info(
                "Connecting to ChromaDB at %s:%s",
                self.config.EMBEDDING_DB_HOST,
                self.config.EMBEDDING_DB_PORT
            )
            self.client = chromadb.HttpClient(
                host=self.config.EMBEDDING_DB_HOST,
                port=self.config.EMBEDDING_DB_PORT or 8000,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.PersistentClient(
                path=f"{self.config.STORAGE_PATH}/{self.config.EMBEDDING_DB_FOLDER}",
                settings=Settings(anonymized_telemetry=False),
            )
        self.embedding_function = (
            self.config.EMBEDDING_DB_FUNCTION
            or embedding_functions.DefaultEmbeddingFunction()
        )

    @classmethod
    def _wrap_results(cls, results) -> list[str | SearchResult]:
        return SearchResults(
            [
                SearchResult(
                    results["documents"][0][i],
                    dict(
                        metadata=results["metadatas"][0][i] or {},
                        id=results["ids"][0][i],
                        distance=results["distances"][0][i],
                    ),
                )
                for i in range(len(results["documents"][0]))
            ]
        )

    def search(
        self,
        collection: str,
        query: str | list,
        n_results: int = 5,
        where: dict = None,
        **kwargs,
    ) -> list[str | SearchResult]:
        if not self.collection_exists(collection):
            return SearchResults([])

        if isinstance(query, str):
            query = [query]

        d = self._get_collection(collection).query(
            query_texts=query, n_results=n_results, where=where, **kwargs
        )
        return (
            self._wrap_results(d)
            if d and d.get("documents") and d["documents"][0]
            else SearchResults([])
        )

    def save_many(self, collection: str, items: list[tuple[str, dict] | str]):
        unique = not self.config.EMBEDDING_DB_ALLOW_DUPLICATES
        texts, ids, metadatas = [], [], []
        for i in items:
            if isinstance(i, str):
                text = i
                metadata = None
            else:
                text = i[0]
                metadata = i[1] or None
            if unique and text in texts:
                continue
            texts.append(text)
            metadatas.append(metadata)
            ids.append(str(hash(text)) if unique else str(uuid.uuid4()))
        self._get_collection(collection, create=True).upsert(
            documents=texts, ids=ids, metadatas=metadatas
        )

    def clear(self, collection: str):
        try:
            self.client.delete_collection(collection)
        except (ValueError, ChromaError):
            pass

    def count(self, collection: str) -> int:
        chroma_collection = self._get_collection(collection)
        return chroma_collection.count() if chroma_collection else 0

    def delete(self, collection: str, what: str | list[str] | dict):
        # pylint: disable=R0801, duplicate-code
        if not self.collection_exists(collection):
            return

        if isinstance(what, str):
            ids, where = [what], None
        elif isinstance(what, list):
            ids, where = what, None
        elif isinstance(what, dict):
            ids, where = None, what
        else:
            raise ValueError("Invalid `what` argument")
        self._get_collection(collection).delete(ids=ids, where=where)

    def get(
        self,
        collection: str,
        ids: list[str] | str = None,
        limit: int = None,
        offset: int = None,
        where: dict = None,
        **kwargs,
    ) -> list[str | SearchResult] | str | SearchResult | None:
        if not self.collection_exists(collection):
            return SearchResults([]) if not isinstance(ids, str) else None

        results = self._get_collection(collection).get(
            ids=[ids] if isinstance(ids, str) else ids,
            limit=limit,
            offset=offset,
            where=where,
            **kwargs,
        )
        search_results = [
            SearchResult(
                results["documents"][i],
                {
                    "metadata": results["metadatas"][i] or {},
                    "id": results["ids"][i],
                },
            )
            for i in range(len(results["documents"]))
        ]
        if isinstance(ids, str):
            return search_results[0] if search_results else None
        return SearchResults(search_results)

    def get_all(self, collection: str) -> list[str | SearchResult]:
        if not self.collection_exists(collection):
            return SearchResults([])
        results = self._get_collection(collection).get()
        return SearchResults(
            [
                SearchResult(
                    results["documents"][i],
                    {
                        "metadata": results["metadatas"][i] or {},
                        "id": results["ids"][i],
                    },
                )
                for i in range(len(results["documents"]))
            ]
        )

    def collection_exists(self, collection: str) -> bool:
        return self._get_collection(collection) is not None

    def _get_collection(self, name: str, create: bool = False):
        if create:
            return self.client.get_or_create_collection(
                name=name, embedding_function=self.embedding_function
            )
        try:
            return self.client.get_collection(
                name, embedding_function=self.embedding_function
            )
        except (ValueError, ChromaError):
            return None
