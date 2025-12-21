# File cache

A lightweight general purpose file-based caching system for Python objetcs designed for cross-cutting concern within the ai-microcore horizontal functionality.  

Target applications: rapid prototyping, R&D, and AI experiments.  

## Overview
The **microcore.file_cache** submodule provides a simple yet powerful way to persist Python objects to disk using pickle serialization. Built on top of microcore's file storage submodule, it offers automatic key generation, hierarchical organization, and straightforward cache management.

## Key Features

- Automatic Key Generation: Creates unique cache keys based on function arguments and current configuration.
- Hierarchical Organization: Organize cached objects using prefix-based subdirectories
- Type Agnostic: Cache any picklable Python object
- Simple API: Intuitive functions for common cache operations
- Configuration Awareness: Automatically incorporates environment config into cache keys

## Use Cases

- Caching expensive computation results
- Storing intermediate AI model outputs
- Persisting experimental data during R&D
- Rapid prototyping with persistent state

> **Note:**
> This implementation prioritizes simplicity and ease of use. For production workloads with high-frequency access patterns, consider faster in-memory storage solutions like Redis or Memcached.


## API Reference

### `cache_dir(prefix: str = "") -> str`

Returns the relative path to the cache directory, optionally with a prefix subdirectory.

```python
from microcore.file_cache import cache_dir
# Base cache directory
path = cache_dir()  # "cache/"

# With prefix
path = cache_dir("llm-requests")  # "cache/llm-requests/"
```

### `build_cache_name(*args, prefix: str = "", **kwargs) -> str`

Generates a unique cache key based on provided arguments and current configuration.

```python
from microcore.file_cache import build_cache_name
key = build_cache_name("my-query", model="gpt-4", prefix="llm-requests")
```

**How it works**: Combines environment config, prefix, args, and kwargs into a dictionary, serializes it, and generates a SHA-256 hash as the filename.

### `cache_hit(cache_name: str) -> bool`

Checks if a cached object exists.

```python
from microcore.file_cache import cache_hit, build_cache_name

key = build_cache_name("data", version=2)
if cache_hit(key):
    print("Cache exists!")
```

### `load_cache(cache_name: str) -> Any`

Retrieves and deserializes a cached Python object.

```python
from microcore.file_cache import load_cache, build_cache_name, cache_hit

key = build_cache_name("results", experiment="A")
if cache_hit(key):
    results = load_cache(key)
```

### `save_cache(cache_name: str, data: Any) -> None`

Serializes and stores a Python object in the cache.

```python
from microcore.file_cache import save_cache, build_cache_name

results = {"accuracy": 0.95, "loss": 0.05}
key = build_cache_name("training", model="v2")
save_cache(key, results)
```

### `delete_cache(cache_name: str) -> bool`

Removes a specific cached object.

```python
from microcore.file_cache import delete_cache, build_cache_name

key = build_cache_name("temp_data")
deleted = delete_cache(key)
```

### `flush_cache(prefix: str = "") -> bool`

Deletes all cached objects, optionally filtered by prefix.

```python
from microcore.file_cache import flush_cache

# Delete all cache
flush_cache()

# Delete only caches with specific prefix
flush_cache("experiments")
```

> **Warning**: Cache flushing affects all modules using the cache. Use prefixes to isolate cache spaces when working in shared environments.

## Usage Examples

### Basic Caching Pattern

```python
from microcore.file_cache import build_cache_name, cache_hit, load_cache, save_cache

def expensive_computation(param1, param2):
    key = build_cache_name(param1, param2, prefix="computations")
    
    if cache_hit(key):
        return load_cache(key)
    
    # Perform expensive operation
    result = perform_computation(param1, param2)
    
    save_cache(key, result)
    return result
```