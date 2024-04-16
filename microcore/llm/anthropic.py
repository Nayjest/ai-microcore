import asyncio
import anthropic
from anthropic.types import ContentBlockDeltaEvent

from ..configuration import Config
from .._prepare_llm_args import prepare_chat_messages
from ..message_types import Role
from ..types import LLMAsyncFunctionType, LLMFunctionType
from ..wrappers.llm_response_wrapper import LLMResponse


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
    args = {"max_tokens": 1024, **config.LLM_DEFAULT_ARGS, **kwargs}
    args["model"] = args.get("model", config.MODEL)
    args.pop("seed", None)  # Not supported by Anthropic
    callbacks: list[callable] = args.pop("callbacks", [])
    if "callback" in args:
        cb = args.pop("callback")
        if cb:
            callbacks.append(cb)
    args["stream"] = bool(callbacks)
    return args, {"callbacks": callbacks}


def _extract_sys_msg(prepared_messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Anthropic does not support system messages,
    so we need to extract them to pass as a separate argument
    """
    system = "\n".join(
        i["content"] for i in prepared_messages if i.get("role") == Role.SYSTEM
    )
    return system, [i for i in prepared_messages if i.get("role") != Role.SYSTEM]


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
            prepare_chat_messages(prompt)
        )
        response = await async_client.messages.create(**args)
        if args["stream"]:
            return await _a_process_streamed_response(response, options["callbacks"])

        for cb in options["callbacks"]:
            cb(response.content[0].text)
        return LLMResponse(response.content[0].text, response.__dict__)

    def llm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        args["system"], args["messages"] = _extract_sys_msg(
            prepare_chat_messages(prompt)
        )
        response = sync_client.messages.create(**args)
        if args["stream"]:
            return _process_streamed_response(response, options["callbacks"])

        for cb in options["callbacks"]:
            cb(response.content[0].text)
        return LLMResponse(response.content[0].text, response.__dict__)

    return llm, allm
