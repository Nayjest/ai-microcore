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
            if key in usage:
                return usage[key]
        return None
    for key in keys:
        val = getattr(usage, key, None)
        if val is not None:
            return val
    return None


def normalize_usage(usage: Any) -> dict | None:
    """Normalize provider-specific usage to a common dict for downstream logging."""
    if usage is None:
        return None

    prompt = _usage_field(
        usage,
        "prompt_tokens",
        "input_tokens",
        "prompt_token_count",
    )
    completion = _usage_field(
        usage,
        "completion_tokens",
        "output_tokens",
        "candidates_token_count",
    )
    total = _usage_field(usage, "total_tokens", "total_token_count")
    cache_read = _usage_field(usage, "cache_read_input_tokens")
    cache_creation = _usage_field(usage, "cache_creation_input_tokens")

    if (
        prompt is None
        and completion is None
        and total is None
        and cache_read is None
        and cache_creation is None
    ):
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
    if cache_creation is not None:
        result["cache_creation_input_tokens"] = cache_creation
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
