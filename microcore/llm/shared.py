import re
from typing import Any

from ..images import FileImage, Image
from ..wrappers.llm_response_wrapper import ImageGenerationResponse, StoredImageGenerationResponse
from ..configuration import Config


def make_remove_hidden_output(config: Config) -> callable:
    pattern = re.compile(
        f"{config.HIDDEN_OUTPUT_BEGIN}.*?{config.HIDDEN_OUTPUT_END}", flags=re.DOTALL
    )

    def remove_hidden_output(text: str) -> str:
        return pattern.sub("", text)

    return remove_hidden_output


def prepare_callbacks(config: Config, args, set_stream: bool = True) -> list[callable]:
    callbacks = (args.pop("callbacks", []) or []) + (config.CALLBACKS or [])
    if "callback" in args:
        cb = args.pop("callback")
        if cb:
            callbacks.append(cb)
    if set_stream and "stream" not in args:
        args["stream"] = bool(callbacks)

    return callbacks


def _usage_field(usage: Any, *keys: str):
    if isinstance(usage, dict):
        for key in keys:
            if usage.get(key) is not None:
                return usage[key]
        return None
    for key in keys:
        val = getattr(usage, key, None)
        if val is not None:
            return val
    return None


def _usage_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_usage_field(usage: Any, *path: str):
    """Read usage['a']['b'] or usage.a.b; missing intermediate → None."""
    cur = usage
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur


def _first_int(*values: Any) -> int | None:
    for value in values:
        n = _usage_int(value)
        if n is not None:
            return n
    return None


def _usage_cache_and_reasoning(usage: Any) -> tuple[int | None, int | None, int | None, bool]:
    """cache_read, cache_write, reasoning, and whether top-level Anthropic cache keys exist."""
    anthropic_read = _first_int(_usage_field(usage, "cache_read_input_tokens"))
    anthropic_write = _first_int(_usage_field(usage, "cache_creation_input_tokens"))
    cache_read = _first_int(
        _nested_usage_field(usage, "prompt_tokens_details", "cached_tokens"),
        _nested_usage_field(usage, "input_tokens_details", "cached_tokens"),
        anthropic_read,
        _usage_field(usage, "cached_content_token_count"),
        _usage_field(usage, "cachedContentTokenCount"),
    )
    cache_write = _first_int(
        _nested_usage_field(usage, "prompt_tokens_details", "cache_write_tokens"),
        _nested_usage_field(usage, "input_tokens_details", "cache_write_tokens"),
        anthropic_write,
    )
    reasoning = _first_int(
        _nested_usage_field(usage, "completion_tokens_details", "reasoning_tokens"),
        _nested_usage_field(usage, "output_tokens_details", "reasoning_tokens"),
        _usage_field(usage, "thoughts_token_count"),
        _usage_field(usage, "thoughtsTokenCount"),
        _usage_field(usage, "reasoning_tokens"),
    )
    anthropic_cache = anthropic_read is not None or anthropic_write is not None
    return cache_read, cache_write, reasoning, anthropic_cache


def _usage_prompt_completion_total(usage: Any) -> tuple[int | None, int | None, int | None, bool]:
    """prompt, completion, total, and whether prompt came from OpenAI-style keys.

    (i.e. ``prompt_tokens`` / ``prompt_token_count``, not ``input_tokens``)
    """
    prompt_openai = _first_int(
        _usage_field(usage, "prompt_tokens", "prompt_token_count", "promptTokenCount"),
    )
    input_tokens = _first_int(_usage_field(usage, "input_tokens"))
    prompt = prompt_openai if prompt_openai is not None else input_tokens
    completion = _first_int(
        _usage_field(
            usage,
            "completion_tokens",
            "output_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
        ),
    )
    total = _first_int(
        _usage_field(usage, "total_tokens", "total_token_count", "totalTokenCount"),
    )
    return prompt, completion, total, prompt_openai is not None


def normalize_usage(usage: Any) -> dict | None:
    """Normalize provider-specific usage to a common dict for downstream logging.

    Always maps prompt / completion / total. When the provider reports them,
    also surfaces optional breakdown fields:

    - ``cache_read_input_tokens`` — OpenAI ``*_details.cached_tokens``,
      Anthropic ``cache_read_input_tokens``, Gemini ``cached_content_token_count`` /
      ``cachedContentTokenCount``
    - ``cache_creation_input_tokens`` — OpenAI ``*_details.cache_write_tokens``,
      Anthropic ``cache_creation_input_tokens``
    - ``reasoning_tokens`` — OpenAI ``*_details.reasoning_tokens``,
      Gemini ``thoughts_token_count`` / ``thoughtsTokenCount``

    Extra fields are additive only: ``prompt_tokens`` / ``completion_tokens`` keep
    each provider's usual totals (no re-bucketing). When cache fields are present,
    ``cache_included_in_prompt`` is True if cache is already part of prompt
    (OpenAI/Gemini) and False when cache sits outside input (Anthropic). If the
    flag is already set (re-normalizing a prior result), it is preserved.
    """
    if usage is None:
        return None

    existing_cache_included = _usage_field(usage, "cache_included_in_prompt")
    cache_read, cache_write, reasoning, anthropic_cache = _usage_cache_and_reasoning(usage)
    prompt, completion, total, openai_style_prompt = _usage_prompt_completion_total(usage)

    counts = (prompt, completion, total, cache_read, cache_write, reasoning)
    if all(value is None for value in counts):
        return None

    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)

    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    if cache_read is not None:
        result["cache_read_input_tokens"] = cache_read
    if cache_write is not None:
        result["cache_creation_input_tokens"] = cache_write
    if reasoning is not None:
        result["reasoning_tokens"] = reasoning
    # Anthropic reports cache outside input_tokens; OpenAI/Gemini nest cache inside prompt.
    # Downstream cost splitters need this to know whether to fold cache into prompt.
    # Preserve a prior flag so re-normalizing an already-normalized dict is idempotent
    # (after the first pass Anthropic looks like OpenAI: prompt_tokens + cache_*).
    if cache_read is not None or cache_write is not None:
        if existing_cache_included is not None:
            result["cache_included_in_prompt"] = bool(existing_cache_included)
        else:
            # Raw Anthropic: input_tokens + top-level cache_* (no prompt_tokens yet).
            result["cache_included_in_prompt"] = (
                openai_style_prompt or prompt is None or not anthropic_cache
            )
    return result


def streaming_usage_attrs(usage: Any) -> dict:
    if normalized := normalize_usage(usage):
        return {"usage": normalized}
    return {}


def attrs_with_normalized_usage(attrs: dict) -> dict:
    result = dict(attrs)
    if "usage" in result:
        if normalized := normalize_usage(result["usage"]):
            result["usage"] = normalized
        else:
            result.pop("usage", None)
    return result


def ensure_stream_include_usage(args: dict) -> None:
    """OpenAI/Azure emit usage in streaming responses only when requested explicitly."""
    stream_options = dict(args.get("stream_options") or {})
    stream_options.setdefault("include_usage", True)
    args["stream_options"] = stream_options


def make_image_generation_response(
    images: list[Image],
    save: str | bool,
    attrs: dict
) -> ImageGenerationResponse | StoredImageGenerationResponse:
    if save:
        file_name = save if isinstance(save, str) else "generated_images/image-<n>.png"
        images = [FileImage(i.store(file_name)) for i in images]
        return StoredImageGenerationResponse(images=images, **attrs)
    return ImageGenerationResponse(images=images, **attrs)
