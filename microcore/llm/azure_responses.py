"""
OpenAI Responses API support (used e.g. for Azure GPT-5.6 reasoning models).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Iterator

from .._prepare_llm_args import prompt_to_message_dicts
from ..configuration import Config, LLMApiBaseError, LLMApiKeyError
from ..llm_backends import ApiType
from ..message_types import Role, ToolCall
from ..types import TPrompt
from .shared import resolve_finish_reason
from .tool_conversion import convert_tool_definitions, openai_responses_tool

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


def should_use_responses_api(config: Config, override: bool | None = None) -> bool:
    """
    Resolve whether the Responses API should be used for a request.

    Priority: per-request ``override`` > ``config.LLM_USE_RESPONSES_API``.
    Defaults to Chat Completions (``False``) when neither is set; the caller/app
    is responsible for opting in (e.g. GPT-5.6 with tools requires the Responses API).
    """
    if override is not None:
        return bool(override)
    return bool(config.LLM_USE_RESPONSES_API)


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

    neutral_messages = prompt_to_message_dicts(prompt)
    converted_messages = convert_to_messages(prompt)
    messages = [
        neutral
        if isinstance(neutral, dict)
        and any(
            key in neutral
            for key in ("_provider_data", "tool_calls", "tool_call")
        )
        else converted
        for neutral, converted in zip(neutral_messages, converted_messages)
    ]
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role is None:
            continue
        provider_data = message.get("_provider_data")
        if (
            role == Role.ASSISTANT
            and message.get("_api_type") == ApiType.OPENAI
            and getattr(provider_data, "output", None) is not None
        ):
            for item in provider_data.output:
                if hasattr(item, "model_dump"):
                    result.append(item.model_dump(exclude_none=True))
                elif isinstance(item, dict):
                    result.append(item)
            continue
        if role == Role.ASSISTANT and message.get("tool_calls"):
            if message.get("content"):
                result.append(
                    {
                        "type": "message",
                        "role": Role.ASSISTANT,
                        "content": message["content"],
                    }
                )
            result.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments or "{}",
                }
                for call in message["tool_calls"]
            )
            continue
        if role == Role.TOOL and (call := message.get("tool_call")):
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": call.id,
                    "output": message.get("content", ""),
                }
            )
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
    if "tools" in responses_args:
        responses_args["tools"] = convert_tool_definitions(
            responses_args["tools"], openai_responses_tool
        )
    return responses_args


def extract_responses_text(response: Any) -> str:
    return str(getattr(response, "output_text", None) or "")


def extract_responses_tool_calls(response: Any) -> list[ToolCall]:
    return [
        ToolCall(name=item.name, arguments=item.arguments, id=item.call_id)
        for item in getattr(response, "output", None) or []
        if getattr(item, "type", None) == "function_call"
    ]


def responses_finish_reason(response: Any) -> str | None:
    """Finish reason of a response in Chat Completions terms."""
    status = getattr(response, "status", None)
    if status == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        return "length" if reason == "max_output_tokens" else reason or status
    return "stop" if status == "completed" else status


def responses_result_attrs(response: Any) -> dict[str, Any]:
    """Tool calls and finish reason of a response, as LLMResponse attributes."""
    tool_calls = extract_responses_tool_calls(response)
    return {
        "tool_calls": tool_calls,
        "finish_reason": resolve_finish_reason(responses_finish_reason(response), tool_calls),
    }


def _chunk(finish_reason: str | None = None, usage: Any = None, **delta) -> SimpleNamespace:
    """A stream chunk in the shape of a Chat Completions chunk."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(**delta), finish_reason=finish_reason)],
        usage=usage,
    )


def _tool_call_chunk(
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    return _chunk(tool_calls=[SimpleNamespace(index=index, id=call_id, function=function)])


def _responses_event_to_chunk(event: Any) -> SimpleNamespace | None:
    """
    Converts a Responses API stream event to the shape of a Chat Completions chunk,
    as far as `_StreamReader` in openai.py reads it.

    A tool call is keyed by its output index: unique per call, but not counted from zero.
    The finish reason is given without regard to tool calls; the reader resolves it
    once the calls are collected.
    """
    event_type = getattr(event, "type", None)
    if event_type == "response.output_text.delta":
        text = str(getattr(event, "delta", None) or "")
        return _chunk(content=text) if text else None
    if event_type == "response.output_item.added":
        item = event.item
        if getattr(item, "type", None) != "function_call":
            return None
        return _tool_call_chunk(event.output_index, item.call_id, item.name, item.arguments)
    if event_type == "response.function_call_arguments.delta":
        return _tool_call_chunk(event.output_index, arguments=event.delta)
    if event_type in ("response.completed", "response.incomplete"):
        response = getattr(event, "response", None)
        return _chunk(
            finish_reason=responses_finish_reason(response),
            usage=getattr(response, "usage", None),
        )
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
) -> dict[str, Any]:
    responses_args = prepare_responses_args(args)
    responses_args["input"] = prompt_to_responses_input(prompt, convert_to_messages)
    return responses_args
