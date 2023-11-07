from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from ..config import Config
from .. import SearchResult, AbstractEmbeddingDB


@dataclass
class ChromaEmbeddingDB(AbstractEmbeddingDB):
    config: Config
    embedding_function: embedding_functions.EmbeddingFunction = None
    client: chromadb.Client = None

    def __post_init__(self):
        self.client = chromadb.PersistentClient(
            path=f"{self.config.STORAGE_PATH}/{self.config.EMBEDDING_DB_FOLDER}"
        )
        self.embedding_function = (
            self.config.EMBEDDING_DB_FUNCTION
            or embedding_functions.DefaultEmbeddingFunction()
        )

    @classmethod
    def _wrap_results(cls, results) -> list[str | SearchResult]:
        return [
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

    def search(
        self,
        collection: str,
        query: str | list,
        n_results: int = 5,
        where: dict = None,
        **kwargs,
    ) -> list[str | SearchResult]:
        try:
            chroma_collection = self.client.get_collection(collection)
        except ValueError:
            return []

        if isinstance(query, str):
            query = [query]

        d = chroma_collection.query(
            query_texts=query, n_results=n_results, where=where, **kwargs
        )
        return (
            self._wrap_results(d)
            if d and d.get("documents") and d["documents"][0]
            else []
        )

    def save_many(self, collection: str, items: list[tuple[str, dict] | str]):
        chroma_collection = self.client.get_or_create_collection(
            name=collection, embedding_function=self.embedding_function
        )
        texts = [i if isinstance(i, str) else i[0] for i in items]
        ids = [str(hash(t)) for t in texts]
        metadatas = [None if isinstance(i, str) else i[1] or None for i in items]
        chroma_collection.upsert(documents=texts, ids=ids, metadatas=metadatas)

    def clear(self, collection: str):
        try:
            self.client.delete_collection(collection)
        except ValueError:
            pass

    def get_all(self, collection: str) -> list[str | SearchResult]:
        try:
            chroma_collection = self.client.get_collection(collection)
        except ValueError:
            return []
        results = chroma_collection.get()
        return [
            SearchResult(
                results["documents"][i],
                {"metadata": results["metadatas"][i] or {}, "id": results["ids"][i]},
            )
            for i in range(len(results["documents"][0]))
        ]
