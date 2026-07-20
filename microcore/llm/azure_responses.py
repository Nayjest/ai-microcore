"""
Azure OpenAI Responses API support for GPT-5.6 reasoning models.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Iterator

from ..configuration import Config, LLMApiBaseError, LLMApiKeyError
from ..llm_backends import ApiPlatform
from ..types import TPrompt

_AZURE_GPT_56_RE = re.compile(r"^gpt-5\.6(?:-|$)", re.IGNORECASE)

_RESPONSES_EXCLUDED_ARGS = frozenset({
    "messages",
    "stream_options",
    "seed",
    "reasoning_effort",
    "callback",
    "callbacks",
    "deployment_id",
    "extra_body",
})


def is_azure_gpt56_model(platform: str | ApiPlatform | None, model: str) -> bool:
    if str(platform or "").strip().lower() != ApiPlatform.AZURE:
        return False
    return _AZURE_GPT_56_RE.match(str(model or "").strip()) is not None


def should_use_azure_responses(config: Config) -> bool:
    return is_azure_gpt56_model(config.LLM_API_PLATFORM, config.MODEL)


def responses_base_url(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        raise LLMApiBaseError(
            "API Base URL is missing. Please enter a valid API Base URL."
        )

    lower = endpoint.lower()
    if lower.endswith("/openai/v1"):
        return endpoint + "/"

    deployments_idx = lower.find("/openai/deployments/")
    if deployments_idx != -1:
        endpoint = endpoint[:deployments_idx].rstrip("/")
        lower = endpoint.lower()

    if lower.endswith("/openai"):
        return endpoint + "/v1/"

    return endpoint + "/openai/v1/"


def _responses_api_key(
    config: Config,
    http_headers: dict[str, Any],
    client_params: dict[str, Any],
) -> str:
    init_key = client_params.get("api_key")
    if init_key is not None and not callable(init_key):
        key = str(init_key).strip()
        if key:
            return key

    key = str(config.LLM_API_KEY or "").strip()
    if key:
        return key

    for header_name, header_value in http_headers.items():
        if header_name.lower() == "api-key" and header_value:
            return str(header_value).strip()
    return ""


def build_responses_client_params(
    config: Config,
    *,
    entra_token_provider: Callable[[], str] | None = None,
) -> dict[str, Any]:
    endpoint = str(config.LLM_API_BASE or "").strip()
    client_params: dict[str, Any] = {
        "base_url": responses_base_url(endpoint),
        **{
            key: value
            for key, value in (config.INIT_PARAMS or {}).items()
            if key not in {"azure_endpoint", "api_version", "azure_ad_token_provider"}
        },
    }

    http_headers = dict(config.HTTP_HEADERS or {})
    if entra_token_provider is not None:
        client_params["api_key"] = entra_token_provider
    else:
        key = _responses_api_key(config, http_headers, client_params)
        if not key:
            raise LLMApiKeyError(
                "API Key is missing. Please enter a valid API Key."
            )
        client_params["api_key"] = key

    if http_headers:
        client_params.setdefault("default_headers", {}).update(http_headers)

    return client_params


def prompt_to_responses_input(
    prompt: TPrompt,
    convert_to_messages: Callable[[TPrompt], list[dict[str, Any]]],
) -> str | list[dict[str, Any]]:
    if isinstance(prompt, str):
        return prompt

    messages = convert_to_messages(prompt)
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role is None:
            continue
        result.append(
            {
                "type": "message",
                "role": role,
                "content": message.get("content", ""),
            }
        )
    if not result:
        return ""
    if len(result) == 1 and result[0]["role"] == "user":
        content = result[0]["content"]
        if isinstance(content, str):
            return content
    return result


def prepare_responses_args(args: dict[str, Any]) -> dict[str, Any]:
    responses_args = {
        key: value
        for key, value in args.items()
        if key not in _RESPONSES_EXCLUDED_ARGS
    }
    reasoning_effort = args.get("reasoning_effort", "medium")
    responses_args["reasoning"] = {"effort": reasoning_effort}
    responses_args["store"] = False
    responses_args["include"] = ["reasoning.encrypted_content"]
    return responses_args


def extract_responses_text(response: Any) -> str:
    return str(getattr(response, "output_text", None) or "")


def get_responses_stream_delta(event: Any) -> str:
    if getattr(event, "type", None) != "response.output_text.delta":
        return ""
    return str(getattr(event, "delta", None) or "")


def get_responses_stream_usage(event: Any) -> Any | None:
    if getattr(event, "type", None) != "response.completed":
        return None
    response = getattr(event, "response", None)
    return getattr(response, "usage", None) if response is not None else None


def _responses_event_to_chunk(event: Any) -> SimpleNamespace | None:
    if text := get_responses_stream_delta(event):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
        )
    if usage := get_responses_stream_usage(event):
        return SimpleNamespace(usage=usage, choices=[])
    return None


def adapt_responses_events(events) -> Iterator[SimpleNamespace]:
    for event in events:
        if chunk := _responses_event_to_chunk(event):
            yield chunk


async def adapt_responses_events_async(events) -> AsyncIterator[SimpleNamespace]:
    async for event in events:
        if chunk := _responses_event_to_chunk(event):
            yield chunk


def build_responses_request(
    prompt: TPrompt,
    convert_to_messages: Callable[[TPrompt], list[dict[str, Any]]],
    args: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    responses_args = prepare_responses_args(args)
    responses_args["input"] = prompt_to_responses_input(prompt, convert_to_messages)
    return responses_args
