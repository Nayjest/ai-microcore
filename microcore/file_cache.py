"""
A lightweight general-purpose file-based caching system for Python objects
designed for cross-cutting concern within the ai-microcore horizontal functionality.

Target applications: rapid prototyping, R&D, and AI experiments.

> **Note:**
> This implementation prioritizes simplicity and ease of use.
> For production workloads with high-frequency access patterns,
> consider faster in-memory storage solutions like Redis or Memcached.
"""
import hashlib
import logging
import pickle
from dataclasses import asdict
from typing import Any

from ._env import config
from .file_storage import storage

CACHE_ROOT_FOLDER = "cache"  # Folder within the storage root


def cache_dir(prefix: str = "") -> str:
    """
    Returns relative path to cache directory with optional prefix used as subfolder
    relative to the file storage root folder.
    Args:
        prefix (str): Optional prefix for subdirectory within the cache folder.
    Returns:
        str: path to the cache directory within the file storage.
    """
    if prefix:  # prevent directory traversal attacks
        prefix = prefix.replace("..", "").strip("/")
    return CACHE_ROOT_FOLDER + "/" + (f"{prefix}/" if prefix else '')


def build_cache_name(*args, prefix: str = "", **kwargs) -> str:
    """
    Build a unique (key) name for cached object based on the provided arguments.
    Returned value serves as both unique key and file name for storing and retrieving cached object.
    """
    obj = {
        "config": asdict(config()),
        "prefix": prefix,
        "kwargs": kwargs,
        "args": args
    }
    serialized = pickle.dumps(obj)
    return cache_dir(prefix) + hashlib.sha256(serialized).hexdigest() + ".pkl"


def cache_hit(cache_name: str) -> bool:
    """
    Check if cache with given name exists.
    Args:
        cache_name (str): Name of cacheable object built with `build_cache_name()` function.
    Returns:
        bool: True if cache exists, False otherwise.
    """
    return storage.exists(cache_name)


def load_cache(cache_name: str) -> Any:
    """
    Load python object from cache under given name built with `build_cache_name()` function.
    """
    logging.debug("Loading cache object \"%s\"", cache_name)
    return pickle.loads(storage.read(cache_name, binary=True))


def save_cache(cache_name: str, data: Any) -> None:
    """
    Save python object to cache under given name built with `build_cache_name()` function.
    """
    logging.debug("Saving cache object \"%s\"", cache_name)
    storage.write(cache_name, pickle.dumps(data))


def delete_cache(cache_name: str) -> bool:
    """
    Delete cached object with given name built with `build_cache_name()` function.
    """
    logging.debug("Deleting cache object \"%s\"", cache_name)
    return storage.delete(cache_name)


def flush_cache(prefix: str = "") -> bool:
    """
    Flush all cached objects within the cache folder or its subfolder defined by prefix.

    Warning:
        Keep in mind that cache can be used in cross-cutting functionality.
        Deleting the whole cache folder may affect other modules relying on cached data.

    Returns:
        bool: True if cache flush operation was successful, False otherwise.
    """
    logging.debug(
        "Flushing cache storage%s",
        " prefixed as \"%s\"" % prefix if prefix else ""
    )
    return storage.delete(cache_dir(prefix))
