import logging
import asyncio
import anthropic
from anthropic.types import ContentBlockDeltaEvent, SignatureDelta, ThinkingDelta

from ..configuration import Config
from .._prepare_llm_args import prompt_to_message_dicts
from ..message_types import Role
from ..types import LLMAsyncFunctionType, LLMFunctionType
from ..wrappers.llm_response_wrapper import LLMResponse
from ..llm_backends import ApiType
from .shared import prepare_callbacks


def _get_response_texts(response, show_thinking: bool = False) -> tuple[str, str]:
    """
    Extract text from response content blocks, preserving their order.
    Returns (response_text, display_text); display_text additionally contains
    thinking blocks wrapped in <think>...</think> when show_thinking is enabled.
    """
    text_parts, display_parts = [], []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
            display_parts.append(block.text)
        elif show_thinking and block_type == "thinking" and block.thinking:
            display_parts.append(f"<think>{block.thinking}</think>")
    response_text = "\n".join(text_parts)
    display_text = "\n\n".join(display_parts) if show_thinking else response_text
    return response_text, display_text


def _get_chunk_text(chunk) -> str:
    if not isinstance(chunk, ContentBlockDeltaEvent):
        return ""
    if isinstance(chunk.delta, ThinkingDelta):
        return ""
    if isinstance(chunk.delta, SignatureDelta):
        return ""
    return chunk.delta.text or ""


def _get_chunk_thinking(chunk) -> str:
    if isinstance(chunk, ContentBlockDeltaEvent) and isinstance(
        chunk.delta, ThinkingDelta
    ):
        return chunk.delta.thinking or ""
    return ""


async def _a_process_streamed_response(
    response, callbacks: list[callable], show_thinking: bool = False
):
    async def send(chunk_text: str):
        for cb in callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(chunk_text)
            else:
                cb(chunk_text)

    response_text: str = ""
    in_thinking = False
    async for chunk in response:
        if show_thinking and (thinking_chunk := _get_chunk_thinking(chunk)):
            if not in_thinking:
                in_thinking = True
                await send("<think>")
            await send(thinking_chunk)
        if text_chunk := _get_chunk_text(chunk):
            if in_thinking:
                in_thinking = False
                await send("</think>\n\n")
            response_text += text_chunk
            await send(text_chunk)
    if in_thinking:
        await send("</think>")
    return LLMResponse(response_text, api_type=ApiType.ANTHROPIC)


def _process_streamed_response(
    response, callbacks: list[callable], show_thinking: bool = False
):
    def send(chunk_text: str):
        [cb(chunk_text) for cb in callbacks]

    response_text: str = ""
    in_thinking = False
    for chunk in response:
        if show_thinking and (thinking_chunk := _get_chunk_thinking(chunk)):
            if not in_thinking:
                in_thinking = True
                send("<think>")
            send(thinking_chunk)
        if text_chunk := _get_chunk_text(chunk):
            if in_thinking:
                in_thinking = False
                send("</think>\n\n")
            response_text += text_chunk
            send(text_chunk)
    if in_thinking:
        send("</think>")
    return LLMResponse(response_text, api_type=ApiType.ANTHROPIC)


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

        response_text, cb_text = _get_response_texts(response, options["show_thinking"])
        for cb in options["callbacks"]:
            if asyncio.iscoroutinefunction(cb):
                await cb(cb_text)
            else:
                cb(cb_text)
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

        response_text, cb_text = _get_response_texts(response, options["show_thinking"])
        for cb in options["callbacks"]:
            cb(cb_text)
        return LLMResponse(
            response_text,
            response.__dict__,
            api_type=ApiType.ANTHROPIC,
            response=response,
        )

    return llm, allm
