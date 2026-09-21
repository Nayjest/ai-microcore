# Token Usage

Every LLM response carries a provider-agnostic `usage` dict with token counts,
so cost tracking, logging and budgeting code stays the same whichever backend is configured.

## Fields

| Field                         | Always | Meaning                                                                                      |
|-------------------------------|:------:|----------------------------------------------------------------------------------------------|
| `prompt_tokens`               |   ✔    | Input tokens                                                                                 |
| `completion_tokens`           |   ✔    | Output tokens (includes reasoning tokens where the provider counts them there)              |
| `total_tokens`                |   ✔    | As reported by the provider, or `prompt_tokens + completion_tokens` when it is omitted       |
| `cache_read_input_tokens`     |        | Prompt tokens served from the provider's prompt cache                                        |
| `cache_creation_input_tokens` |        | Prompt tokens written to the prompt cache by this request                                    |
| `reasoning_tokens`            |        | Thinking / reasoning tokens                                                                  |
| `cache_included_in_prompt`    |        | Present whenever a cache field is present; see [Cost calculation](#cost-calculation)         |

Optional fields are present only when the provider reports them, so use `.get()`.

## Reading usage

### `llm()` / `allm()`

The returned `LLMResponse` is a `str` subclass; `usage` is one of its attributes:

```python
from microcore import llm, allm

response = llm("Hi there")
print(response.usage)
# {'prompt_tokens': 12, 'completion_tokens': 4, 'total_tokens': 16}

response = await allm("Hi there")
print(response.usage["total_tokens"])
```

### Streaming with callbacks

Streaming does not change anything: the response returned after the stream completes
carries the final usage.

```python
response = llm("Hi there", callback=lambda chunk: print(chunk, end=""))
print(response.usage)
```

### `llm_parallel()`

Returns a list of `LLMResponse`; sum over it as needed:

```python
from microcore import llm_parallel

responses = await llm_parallel(["1+1=", "2+2=", "3+3="])
total = sum(r.usage["total_tokens"] for r in responses)
```

### `llm_stream()`

`llm_stream()` yields text chunks and does not return a response object.
Capture usage through an after-handler instead (see below).

### After-handlers

`llm_after_handlers` receive the same `LLMResponse` for every call made through
`llm`, `allm`, `llm_parallel` and `llm_stream`, which makes them the right place
for centralized accounting:

```python
import microcore as mc

spent = {"prompt": 0, "completion": 0}

def track(response):
    usage = getattr(response, "usage", None) or {}
    spent["prompt"] += usage.get("prompt_tokens", 0)
    spent["completion"] += usage.get("completion_tokens", 0)

mc.env().llm_after_handlers.append(track)
```

## Cost calculation

Providers disagree on whether cached tokens are part of `prompt_tokens`:

- **OpenAI, Gemini, DeepSeek** (and other OpenAI-compatible APIs): `prompt_tokens` is the
  full input; cache fields are a breakdown of it → `cache_included_in_prompt` is `True`.
- **Anthropic**: `prompt_tokens` excludes cached tokens; they are billed separately
  → `cache_included_in_prompt` is `False`.

```python
u = response.usage
cache_read = u.get("cache_read_input_tokens", 0)
cache_write = u.get("cache_creation_input_tokens", 0)
if u.get("cache_included_in_prompt", True):
    uncached = u["prompt_tokens"] - cache_read - cache_write
else:
    uncached = u["prompt_tokens"]
cost = (
    uncached * PRICE_INPUT
    + cache_read * PRICE_CACHE_READ
    + cache_write * PRICE_CACHE_WRITE
    + u["completion_tokens"] * PRICE_OUTPUT
)
```

The normalized dict is idempotent: feeding it back into `microcore.llm.shared.normalize_usage`
returns it unchanged, including the flag.

## Provider notes

| Provider  | Source fields                                                                                             | Notes                                                                                      |
|-----------|-----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| OpenAI    | `prompt_tokens_details.cached_tokens` / `.cache_write_tokens`, `completion_tokens_details.reasoning_tokens` | Automatic caching for prompts ≥ 1024 tokens                                                |
| Responses API | `input_tokens_details.cached_tokens`, `output_tokens_details.reasoning_tokens`                        |                                                                                            |
| Anthropic | `cache_read_input_tokens`, `cache_creation_input_tokens`                                                  | Cache is opt-in: mark content blocks with `cache_control`                                  |
| Gemini    | `cached_content_token_count`, `thoughts_token_count`                                                      | Implicit cache reports hits only above a model-specific prefix size (~4k tokens observed)  |
| DeepSeek  | `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens`                       |                                                                                            |
