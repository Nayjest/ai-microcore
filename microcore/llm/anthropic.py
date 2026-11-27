import json
import logging
import inspect
import re
import anthropic
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    TextDelta,
    ThinkingDelta,
)

from ..configuration import Config
from .._prepare_llm_args import prompt_to_message_dicts
from ..message_types import Role, ToolCall
from ..types import LLMAsyncFunctionType, LLMFunctionType
from ..wrappers.llm_response_wrapper import LLMResponse
from ..llm_backends import ApiType
from .shared import (
    attrs_with_normalized_usage,
    prepare_callbacks,
    resolve_finish_reason,
    streaming_usage_attrs,
    ToolCallAccumulator,
)
from .tool_conversion import anthropic_tool, convert_tool_definitions

THINK_OPEN, THINK_CLOSE = "<think>", "</think>\n"
PART_SEPARATOR = "\n"
FINISH_REASONS = {  # Anthropic stop reason: finish reason in Chat Completions terms
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _get_response_text(response, show_thinking: bool = False) -> str:
    """
    Extract text from response content blocks, preserving their order.
    When show_thinking is enabled, thinking blocks are included,
    wrapped in <think>...</think>.
    Output format is identical to the one produced by _StreamFormatter.
    """
    parts = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text" and block.text:
            parts.append(block.text)
        elif show_thinking and block_type == "thinking" and block.thinking:
            parts.append(f"{THINK_OPEN}{block.thinking}{THINK_CLOSE}")
    return PART_SEPARATOR.join(parts)


def _finish_reason(stop_reason: str | None, tool_calls: list[ToolCall]) -> str | None:
    """
    Finish reason in Chat Completions terms.
    Stop reasons without a counterpart (e.g. pause_turn) are returned as they are.
    """
    return resolve_finish_reason(FINISH_REASONS.get(stop_reason, stop_reason), tool_calls)


def _result_attrs(response) -> dict:
    """Tool calls and finish reason of a non-streamed response, as LLMResponse attributes."""
    tool_calls = [
        ToolCall(name=block.name, arguments=json.dumps(block.input), id=block.id)
        for block in response.content
        if getattr(block, "type", None) == "tool_use"
    ]
    return {
        "tool_calls": tool_calls,
        "finish_reason": _finish_reason(getattr(response, "stop_reason", None), tool_calls),
    }


class _StreamFormatter:
    """
    Transforms raw stream events into output chunks,
    formatted identically to _get_response_text():
    thinking blocks (if enabled) are wrapped in <think>...</think>,
    non-empty content blocks are separated with PART_SEPARATOR.
    """

    def __init__(self, show_thinking: bool = False):
        self.show_thinking = show_thinking
        self.is_first_part = True
        self.is_thinking_block = False
        self.opening: str | None = None  # pending output for the current block start
        self.opened = False  # current block has produced output

    def _delta_content(self, chunk) -> str:
        if isinstance(chunk.delta, TextDelta):
            return chunk.delta.text or ""
        if isinstance(chunk.delta, ThinkingDelta) and self.show_thinking:
            return chunk.delta.thinking or ""
        return ""  # tool input, signature, citations

    def process(self, chunk) -> list[str]:
        """Returns output chunks to emit for the given stream event."""
        out = []
        if isinstance(chunk, ContentBlockStartEvent):
            self.is_thinking_block = (
                getattr(chunk.content_block, "type", None) == "thinking"
            )
            self.opened = False
            self.opening = ("" if self.is_first_part else PART_SEPARATOR) + (
                THINK_OPEN if self.is_thinking_block else ""
            )
        elif isinstance(chunk, ContentBlockDeltaEvent):
            if content := self._delta_content(chunk):
                if not self.opened and self.opening is not None:
                    out.append(self.opening)
                    self.opening = None
                    self.opened = True
                    self.is_first_part = False
                out.append(content)
        elif isinstance(chunk, ContentBlockStopEvent):
            out.extend(self.finish())
        return [i for i in out if i]

    def finish(self) -> list[str]:
        """Closes the current block if needed; call at end of block or stream."""
        out = []
        if self.opened and self.is_thinking_block:
            out.append(THINK_CLOSE)
        self.opening = None
        self.opened = False
        self.is_thinking_block = False
        return out


def _update_stream_usage(chunk, usage: dict) -> dict:
    if isinstance(chunk, MessageStartEvent):
        msg_usage = chunk.message.usage
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            val = getattr(msg_usage, key, None)
            if val is not None:
                usage[key] = val
    elif isinstance(chunk, MessageDeltaEvent):
        if chunk.usage and chunk.usage.output_tokens is not None:
            usage["output_tokens"] = chunk.usage.output_tokens
    return usage


class _StreamResult:
    """
    Collects what the final LLMResponse needs besides the text:
    usage, tool calls and finish reason.
    """

    def __init__(self):
        self.usage: dict = {}
        self.tool_calls = ToolCallAccumulator()
        self.stop_reason: str | None = None

    def update(self, chunk) -> None:
        self.usage = _update_stream_usage(chunk, self.usage)
        if isinstance(chunk, ContentBlockStartEvent):
            block = chunk.content_block
            if getattr(block, "type", None) == "tool_use":
                self.tool_calls.add(chunk.index, block.id, block.name)
        elif isinstance(chunk, ContentBlockDeltaEvent):
            # server tools (e.g. web search) stream their input too, but are not tool calls
            if isinstance(chunk.delta, InputJSONDelta) and chunk.index in self.tool_calls:
                self.tool_calls.add(chunk.index, arguments=chunk.delta.partial_json)
        elif isinstance(chunk, MessageDeltaEvent):
            self.stop_reason = chunk.delta.stop_reason or self.stop_reason

    def response(self, text: str) -> LLMResponse:
        tool_calls = self.tool_calls.tool_calls()
        return LLMResponse(
            text,
            attrs=streaming_usage_attrs(self.usage),
            api_type=ApiType.ANTHROPIC,
            tool_calls=tool_calls,
            finish_reason=_finish_reason(self.stop_reason, tool_calls),
        )


async def _a_process_streamed_response(
    response, callbacks: list[callable], show_thinking: bool = False
):
    parts: list[str] = []
    result = _StreamResult()

    async def send(chunk_text: str):
        parts.append(chunk_text)
        for cb in callbacks:
            if inspect.iscoroutinefunction(cb):
                await cb(chunk_text)
            else:
                cb(chunk_text)

    formatter = _StreamFormatter(show_thinking)
    async for chunk in response:
        result.update(chunk)
        for piece in formatter.process(chunk):
            await send(piece)
    for piece in formatter.finish():
        await send(piece)
    return result.response("".join(parts))


def _process_streamed_response(
    response, callbacks: list[callable], show_thinking: bool = False
):
    parts: list[str] = []
    result = _StreamResult()

    def send(chunk_text: str):
        parts.append(chunk_text)
        [cb(chunk_text) for cb in callbacks]

    formatter = _StreamFormatter(show_thinking)
    for chunk in response:
        result.update(chunk)
        for piece in formatter.process(chunk):
            send(piece)
    for piece in formatter.finish():
        send(piece)
    return result.response("".join(parts))


def _default_max_tokens(model: str, stream: bool) -> int:
    """
    `max_tokens` for a request that does not set it.
    Claude models think before they answer, and thinking counts towards `max_tokens`,
    so the default leaves room for both, within the model's output limit.
    Without streaming, the Anthropic SDK refuses a `max_tokens` that could take
    over 10 minutes to generate: above about 21,333, or 8,192 for Opus 4.
    """
    model = model.lower().replace(".", "-")  # also "anthropic/claude-3.5-sonnet" etc.
    if "claude" not in model:  # another model served over the Anthropic API
        return 4096
    if re.search(r"claude-(v?[12]|instant|3-(opus|sonnet|haiku))", model):
        return 4096
    if "claude-3-5-" in model:
        return 8192
    if re.search(r"claude-(opus-4|4-opus)(?!-[2-9](\D|$))", model):  # Opus 4, 4.1
        return 32_000 if stream else 8192
    return 64_000 if stream else 20_000


def _accepts_sampling_params(model: str, api_version: str) -> bool:
    """
    Whether a request may carry `temperature`, `top_p` and `top_k`.
    API version 2026-10-21 (Anthropic SDK 2+) removed them for every model;
    earlier versions reject non-default values on Claude 4.7+ and Claude Mythos Preview.
    See https://platform.claude.com/docs/en/about-claude/model-deprecations
    """
    if api_version >= "2026-10-21":
        return False
    model = model.lower().replace(".", "-")  # also "anthropic/claude-opus-4.7" etc.
    if "claude-mythos-preview" in model:
        return False
    # "claude-<family>-<major>[-<minor>]", a date suffix is not a minor version
    version = re.search(r"claude-[a-z]+-(\d+)(?:-(\d)(?!\d))?", model)
    return not version or (int(version[1]), int(version[2] or 0)) < (4, 7)


def _prepare_llm_arguments(config: Config, kwargs: dict, api_version: str):
    args = {**config.LLM_DEFAULT_ARGS, **kwargs}
    args["model"] = args.get("model", config.MODEL)
    # Remove arguments not supported by Anthropic
    args.pop("seed", None)
    args.pop("n", None)
    if not _accepts_sampling_params(args["model"], api_version):
        for name in ("temperature", "top_p", "top_k"):
            args.pop(name, None)
    if "temperature" in args and "top_p" in args:
        del args["top_p"]
        logging.warning(
            "`temperature` and `top_p` cannot both be specified for this model. "
            "`top_p` parameter will be ignored. "
        )
    show_thinking = args.pop("show_thinking", getattr(config, "SHOW_THINKING", False))
    callbacks = prepare_callbacks(config, args)
    if "max_tokens" not in args:
        args["max_tokens"] = _default_max_tokens(args["model"], args.get("stream"))
    if "tools" in args:
        args["tools"] = convert_tool_definitions(args["tools"], anthropic_tool)
    return args, {"callbacks": callbacks, "show_thinking": show_thinking}


def _prepare_messages(prompt) -> list[dict]:
    """Convert portable tool turns to Anthropic content blocks."""
    prepared = []
    for message in prompt_to_message_dicts(prompt):
        role = message.get("role")
        provider_data = message.pop("_provider_data", None)
        api_type = message.pop("_api_type", None)
        tool_calls = message.pop("tool_calls", None)
        tool_call = message.pop("tool_call", None)

        if (
            role == Role.ASSISTANT
            and api_type == ApiType.ANTHROPIC
            and getattr(provider_data, "content", None) is not None
        ):
            prepared.append({"role": Role.ASSISTANT, "content": provider_data.content})
        elif role == Role.ASSISTANT and tool_calls:
            content = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.args,
                }
                for call in tool_calls
            )
            prepared.append({"role": Role.ASSISTANT, "content": content})
        elif role == Role.TOOL and tool_call:
            block = {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": message.get("content", ""),
            }
            previous_content = prepared[-1].get("content") if prepared else None
            if (
                prepared
                and prepared[-1].get("role") == Role.USER
                and isinstance(previous_content, list)
                and all(
                    isinstance(i, dict) and i.get("type") == "tool_result"
                    for i in previous_content
                )
            ):
                previous_content.append(block)
            else:
                prepared.append({"role": Role.USER, "content": [block]})
        else:
            prepared.append(message)
    return prepared


def _extract_sys_msg(prepared_messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Anthropic does not support system messages,
    so we need to extract them to pass as a separate argument.
    Also ensures the first and last messages are from user,
    and there is assistants message between user messages.
    """
    system = "\n".join(
        i["content"] for i in prepared_messages if i.get("role") == Role.SYSTEM
    )
    messages = [i for i in prepared_messages if i.get("role") != Role.SYSTEM]

    empty_user_msg = {"role": Role.USER, "content": "--//--"}
    if not messages or messages[0]["role"] != Role.USER:
        messages.insert(0, empty_user_msg)

    # Ensure proper alternation and last message is from User
    normalized_messages = []
    expected_role = Role.USER
    for msg in messages:
        if msg["role"] == expected_role:
            normalized_messages.append(msg)
            expected_role = Role.ASSISTANT if expected_role == Role.USER else Role.USER
        elif msg["role"] == Role.USER and expected_role == Role.ASSISTANT:
            normalized_messages.append({"role": Role.ASSISTANT, "content": "--//--"})
            normalized_messages.append(msg)
            expected_role = Role.ASSISTANT

    # Ensure the last message is from User
    if normalized_messages[-1]["role"] != Role.USER:
        normalized_messages.append(empty_user_msg)

    return system, normalized_messages


def make_llm_functions(config: Config) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    client_params = {
        "api_key": config.LLM_API_KEY,
        "base_url": config.LLM_API_BASE,
        **config.INIT_PARAMS,
    }
    if config.HTTP_HEADERS:
        if "default_headers" not in client_params:
            client_params["default_headers"] = {}
        client_params["default_headers"].update(config.HTTP_HEADERS)

    sync_client = anthropic.Anthropic(**client_params)
    async_client = anthropic.AsyncAnthropic(**client_params)
    api_version = sync_client.default_headers["anthropic-version"]

    async def allm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs, api_version)
        args["system"], args["messages"] = _extract_sys_msg(
            _prepare_messages(prompt)
        )
        response = await async_client.messages.create(**args)
        if args.get("stream"):
            return await _a_process_streamed_response(
                response, options["callbacks"], options["show_thinking"]
            )

        response_text = _get_response_text(response, options["show_thinking"])
        for cb in options["callbacks"]:
            if inspect.iscoroutinefunction(cb):
                await cb(response_text)
            else:
                cb(response_text)
        return LLMResponse(
            response_text,
            attrs_with_normalized_usage(response.__dict__),
            api_type=ApiType.ANTHROPIC,
            response=response,
            **_result_attrs(response),
        )

    def llm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs, api_version)
        args["system"], args["messages"] = _extract_sys_msg(
            _prepare_messages(prompt)
        )
        response = sync_client.messages.create(**args)
        if args.get("stream"):
            return _process_streamed_response(
                response, options["callbacks"], options["show_thinking"]
            )

        response_text = _get_response_text(response, options["show_thinking"])
        for cb in options["callbacks"]:
            cb(response_text)
        return LLMResponse(
            response_text,
            attrs_with_normalized_usage(response.__dict__),
            api_type=ApiType.ANTHROPIC,
            response=response,
            **_result_attrs(response),
        )

    return llm, allm
