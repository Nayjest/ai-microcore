import logging

import tiktoken
import requests.exceptions
from ._env import env


class CantLoadTikTokenEncoding(RuntimeError):
    ...


def _resolve_tiktoken_encoding(
    for_model: str = None, encoding: str | tiktoken.Encoding = None
) -> tiktoken.Encoding:
    assert (
        for_model is None or encoding is None
    ), "You may specify encoding or for_model(LLM), but not both"
    if isinstance(encoding, tiktoken.Encoding):
        return encoding
    if for_model is None and encoding is None:
        if env().config.TIKTOKEN_ENCODING:
            return _resolve_tiktoken_encoding(encoding=env().config.TIKTOKEN_ENCODING)
        for_model = (
            env().config.LLM_DEFAULT_ARGS.get("model", None) or env().config.MODEL
        )
    if for_model:
        try:
            if for_model.startswith("gpt-4.1") or for_model.startswith("gpt-4.5"):
                return tiktoken.get_encoding("o200k_base")
            return tiktoken.encoding_for_model(for_model)
        except (KeyError, requests.exceptions.ConnectionError):
            logging.warning(
                f"Can't resolve tiktoken encoding for '{for_model}'. "
                f"Default encoding will be used."
            )
    encoding = encoding or "cl100k_base"
    try:
        return tiktoken.get_encoding(encoding)
    except (ValueError, requests.exceptions.ConnectionError) as e:
        raise CantLoadTikTokenEncoding(
            f"Can't load tiktoken encoding '{encoding}'"
        ) from e


def encode(
    string: str, for_model: str = None, encoding: str | tiktoken.Encoding = None
) -> list[int]:
    """Encodes string to LLM tokens"""
    return _resolve_tiktoken_encoding(for_model, encoding).encode(string)


def num_tokens_from_string(
    string: str, for_model: str = None, encoding: str | tiktoken.Encoding = None
) -> int:
    """Returns the number of tokens in a text string."""
    return len(encode(string, for_model=for_model, encoding=encoding))


def fit_to_token_size(
    docs: list[str],
    max_tokens: int,
    min_documents: int = None,
    for_model: str = None,
    encoding: str | tiktoken.Encoding = None,
) -> tuple[list[str], int]:
    """
    Fit the list of documents to the max_tokens size.
    Returns the new list of documents and qty of removed items
    """
    encoding = _resolve_tiktoken_encoding(for_model, encoding)
    tot_size = 0
    for i, doc in enumerate(docs):
        tot_size += num_tokens_from_string(str(doc), encoding=encoding)
        if min_documents and i < min_documents:
            continue
        if tot_size > max_tokens:
            result = docs[:i]
            return result, len(docs) - len(result)
    return docs, 0
