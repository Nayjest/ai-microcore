import logging
import asyncio
import anthropic
from anthropic.types import ContentBlockDeltaEvent

from ..configuration import Config
from .._prepare_llm_args import prompt_to_message_dicts
from ..message_types import Role
from ..types import LLMAsyncFunctionType, LLMFunctionType
from ..wrappers.llm_response_wrapper import LLMResponse
from .shared import prepare_callbacks


def _get_chunk_text(chunk):
    return isinstance(chunk, ContentBlockDeltaEvent) and chunk.delta.text or ""


async def _a_process_streamed_response(response, callbacks: list[callable]):
    response_text: str = ""
    async for chunk in response:
        if text_chunk := _get_chunk_text(chunk):
            response_text += text_chunk
            for cb in callbacks:
                if asyncio.iscoroutinefunction(cb):
                    await cb(text_chunk)
                else:
                    cb(text_chunk)
    return LLMResponse(response_text, {})


def _process_streamed_response(response, callbacks: list[callable]):
    response_text: str = ""
    for chunk in response:
        if text_chunk := _get_chunk_text(chunk):
            response_text += text_chunk
            [cb(text_chunk) for cb in callbacks]
    return LLMResponse(response_text, {})


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
    callbacks = prepare_callbacks(config, args)
    return args, {"callbacks": callbacks}


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
    sync_client = anthropic.Anthropic(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        **config.INIT_PARAMS,
    )
    async_client = anthropic.AsyncAnthropic(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        **config.INIT_PARAMS,
    )

    async def allm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        args["system"], args["messages"] = _extract_sys_msg(
            prompt_to_message_dicts(prompt)
        )
        response = await async_client.messages.create(**args)
        if args.get("stream"):
            return await _a_process_streamed_response(response, options["callbacks"])

        for cb in options["callbacks"]:
            if asyncio.iscoroutinefunction(cb):
                await cb(response.content[0].text)
            else:
                cb(response.content[0].text)
        return LLMResponse(response.content[0].text, response.__dict__)

    def llm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        args["system"], args["messages"] = _extract_sys_msg(
            prompt_to_message_dicts(prompt)
        )
        response = sync_client.messages.create(**args)
        if args.get("stream"):
            return _process_streamed_response(response, options["callbacks"])

        for cb in options["callbacks"]:
            cb(response.content[0].text)
        return LLMResponse(response.content[0].text, response.__dict__)

    return llm, allm
