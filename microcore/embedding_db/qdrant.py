import hashlib
import logging
import sys
from dataclasses import dataclass
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointIdsList,
    CollectionInfo,
    Record,
    FieldCondition,
    Filter,
    MatchValue,
    MatchText
)
from qdrant_client.models import VectorParams, Distance, PointStruct, ScoredPoint

from ..configuration import Config
from .. import SearchResult, SearchResults, AbstractEmbeddingDB


def is_sentence_transformer(fn):
    return fn.__class__.__name__ == 'SentenceTransformer'


def prepare_embedding_function(fn):
    if is_sentence_transformer(fn):
        return lambda x: fn.encode(x).tolist()
    return fn


@dataclass
class QdrantEmbeddingDB(AbstractEmbeddingDB):
    config: Config
    embedding_function: callable = None
    client: QdrantClient = None

    def __post_init__(self):

        logging.info(
            "Connecting to Qdrant at %s:%s",
            self.config.EMBEDDING_DB_HOST,
            self.config.EMBEDDING_DB_PORT
        )
        self.client = QdrantClient(
            host=self.config.EMBEDDING_DB_HOST,
            port=self.config.EMBEDDING_DB_PORT or 6333,
            timeout=self.config.EMBEDDING_DB_TIMEOUT,
        )
        self.embedding_function = prepare_embedding_function(self.config.EMBEDDING_DB_FUNCTION)

    @classmethod
    def _wrap_results(cls, points: list[ScoredPoint | Record]) -> list[str | SearchResult]:
        return SearchResults(
            [
                SearchResult(
                    i.payload["_text"],
                    dict(
                        id=i.id,
                        distance=i.score if hasattr(i, "score") else 0,
                        metadata={k: v for k, v in i.payload.items() if k != "_text"},
                    ),

                )
                for i in points
            ]
        )

    @classmethod
    def _convert_where(  # pylint: disable=too-many-branches
        cls,
        where: dict | None,
        kwargs=None
    ) -> Filter | None:
        where_doc = kwargs and kwargs.get("where_document", {}).get("$contains", None)
        if isinstance(where, Filter):
            if where_doc:
                raise ValueError(
                    "Cannot use `where_document` with Filter object passed as `where` argument. "
                    "Please use a dictionary instead."
                )
            return where

        conditions = []
        _and = True
        if where:
            if "$or" in where:
                _and = False
                for i in where["$or"]:
                    for k, v in i.items():
                        conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            elif "$and" in where:
                _and = True
                for i in where["$and"]:
                    for k, v in i.items():
                        conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            else:
                for k, v in where.items():
                    conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))

        # ChromaDB format
        if where_doc:
            if not _and:
                raise ValueError(
                    "Cannot use `where_document` with `$or` condition. "
                )
            conditions.append(
                FieldCondition(
                    key="_text",
                    match=MatchText(text=kwargs["where_document"]["$contains"])
                )
            )

        if not conditions:
            return None
        if _and:
            return Filter(must=conditions)
        return Filter(should=conditions)

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

        if not isinstance(query, str):
            raise ValueError("`query` must be a string")

        query_vector = self.embedding_function(query)
        where = self._convert_where(where, kwargs)
        kwargs.pop("where_document", None)
        hits = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=n_results,
            query_filter=where,
            **kwargs,
        )
        return self._wrap_results(hits.points)

    def save_many(self, collection: str, items: list[tuple[str, dict] | str]):
        if not self.collection_exists(collection):
            self._create_collection(collection)
        point_structs = []
        ids = set()
        unique = not self.config.EMBEDDING_DB_ALLOW_DUPLICATES
        for i in items:
            if isinstance(i, str):
                text = i
                metadata = dict()
            else:
                text = i[0]
                metadata = i[1] or {}
            metadata["_text"] = text
            if unique:
                new_id = str(uuid.UUID(hashlib.md5(text.encode()).hexdigest()))
                if new_id in ids:
                    continue
                ids.add(new_id)
            else:
                new_id = str(uuid.uuid4())
            point_structs.append(
                PointStruct(
                    id=new_id,
                    vector=self.embedding_function(text),
                    payload=metadata
                )
            )

        operation_info = self.client.upsert(
            collection_name=collection,
            wait=True,
            points=point_structs,
        )
        return operation_info

    def clear(self, collection: str):
        if self.collection_exists(collection):
            self.client.delete_collection(collection_name=collection)

    def count(self, collection: str) -> int:
        if self.collection_exists(collection):
            return self._get_collection(collection).points_count
        return 0

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
        if ids is not None:
            points_selector = PointIdsList(points=ids)
        else:
            points_selector = self._convert_where(where)
        self.client.delete(collection_name=collection, points_selector=points_selector)

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
        if ids:
            raise NotImplementedError("Getting by ids is not supported for Qdrant")
        where = self._convert_where(where, kwargs)
        kwargs.pop("where_document", None)
        search_results = self._wrap_results(
            self.client.scroll(
                collection,
                limit=limit or sys.maxsize - 1,
                offset=offset or 0,
                scroll_filter=where,
                **kwargs
            )[0]
        )
        if isinstance(ids, str):
            return search_results[0] if search_results else None
        return search_results

    def get_all(self, collection: str) -> list[str | SearchResult]:
        if not self.collection_exists(collection):
            return SearchResults([])
        return self._wrap_results(self.client.scroll(collection, limit=sys.maxsize - 1)[0])

    def collection_exists(self, collection: str) -> bool:
        return self.client.collection_exists(collection)

    def _create_collection(self, name: str):
        assert self.config.EMBEDDING_DB_SIZE > 0
        size = self.config.EMBEDDING_DB_SIZE
        distance = Distance.COSINE
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=distance),
        )

    def _get_collection(self, name: str, create: bool = False) -> CollectionInfo | None:
        if not self.collection_exists(name):
            if create:
                self._create_collection(name)
            else:
                return None
        return self.client.get_collection(name)
