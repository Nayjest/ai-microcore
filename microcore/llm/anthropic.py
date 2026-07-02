import logging
import asyncio
import anthropic
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    SignatureDelta,
    ThinkingDelta,
)

from ..configuration import Config
from .._prepare_llm_args import prompt_to_message_dicts
from ..message_types import Role
from ..types import LLMAsyncFunctionType, LLMFunctionType
from ..wrappers.llm_response_wrapper import LLMResponse
from ..llm_backends import ApiType
from .shared import prepare_callbacks

THINK_OPEN, THINK_CLOSE = "<think>", "</think>\n"
PART_SEPARATOR = "\n"


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
        if isinstance(chunk.delta, ThinkingDelta):
            if self.show_thinking:
                return chunk.delta.thinking or ""
            return ""
        if isinstance(chunk.delta, SignatureDelta):
            return ""
        return chunk.delta.text or ""

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


async def _a_process_streamed_response(
    response, callbacks: list[callable], show_thinking: bool = False
):
    parts: list[str] = []

    async def send(chunk_text: str):
        parts.append(chunk_text)
        for cb in callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(chunk_text)
            else:
                cb(chunk_text)

    formatter = _StreamFormatter(show_thinking)
    async for chunk in response:
        for piece in formatter.process(chunk):
            await send(piece)
    for piece in formatter.finish():
        await send(piece)
    return LLMResponse("".join(parts), api_type=ApiType.ANTHROPIC)


def _process_streamed_response(
    response, callbacks: list[callable], show_thinking: bool = False
):
    parts: list[str] = []

    def send(chunk_text: str):
        parts.append(chunk_text)
        [cb(chunk_text) for cb in callbacks]

    formatter = _StreamFormatter(show_thinking)
    for chunk in response:
        for piece in formatter.process(chunk):
            send(piece)
    for piece in formatter.finish():
        send(piece)
    return LLMResponse("".join(parts), api_type=ApiType.ANTHROPIC)


def _prepare_llm_arguments(config: Config, kwargs: dict):
    args = {**config.LLM_DEFAULT_ARGS, **kwargs}
    args["model"] = args.get("model", config.MODEL)
    if "max_tokens" not in args:
        if "claude-3-5-sonnet" in args["model"]:
            args["max_tokens"] = 8192
        elif "claude-3-7-sonnet" in args["model"]:
            args["max_tokens"] = 16384
        else:
            args["max_tokens"] = 4096
    # Remove arguments not supported by Anthropic
    args.pop("seed", None)
    args.pop("n", None)
    if "temperature" in args and "top_p" in args:
        del args["top_p"]
        logging.warning(
            "`temperature` and `top_p` cannot both be specified for this model. "
            "`top_p` parameter will be ignored. "
        )
    show_thinking = args.pop("show_thinking", config.SHOW_THINKING)
    callbacks = prepare_callbacks(config, args)
    return args, {"callbacks": callbacks, "show_thinking": show_thinking}


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

    async def allm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        args["system"], args["messages"] = _extract_sys_msg(
            prompt_to_message_dicts(prompt)
        )
        response = await async_client.messages.create(**args)
        if args.get("stream"):
            return await _a_process_streamed_response(
                response, options["callbacks"], options["show_thinking"]
            )

        response_text = _get_response_text(response, options["show_thinking"])
        for cb in options["callbacks"]:
            if asyncio.iscoroutinefunction(cb):
                await cb(response_text)
            else:
                cb(response_text)
        return LLMResponse(
            response_text,
            response.__dict__,
            api_type=ApiType.ANTHROPIC,
            response=response,
        )

    def llm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        args["system"], args["messages"] = _extract_sys_msg(
            prompt_to_message_dicts(prompt)
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
            response.__dict__,
            api_type=ApiType.ANTHROPIC,
            response=response,
        )

    return llm, allm
