import asyncio
import openai

from ..config import Config, ApiType
from .._prepare_llm_args import prepare_chat_messages, prepare_prompt
from ..types import LLMAsyncFunctionType, LLMFunctionType
from ..wrappers.llm_response_wrapper import LLMResponse
from ..utils import is_chat_model

OPENAI_V1_API = True


def _get_chunk_text(chunk, mode_chat_model: bool):
    # Azure API gives first chunk with empty choices
    if len(chunk.choices) == 0:
        return ""
    choice = chunk.choices[0]
    if mode_chat_model:
        if hasattr(choice, "delta"):
            return getattr(choice.delta, "content", "")
        return ""
    return getattr(choice, "text", "")


async def _a_process_streamed_response(
    response, callbacks: list[callable], chat_model_used: bool
):
    response_text: str = ""
    async for chunk in response:
        if text_chunk := _get_chunk_text(chunk, chat_model_used):
            response_text += text_chunk
            for cb in callbacks:
                if asyncio.iscoroutinefunction(cb):
                    await cb(text_chunk)
                else:
                    cb(text_chunk)
    return LLMResponse(response_text, {})


def _process_streamed_response(
    response, callbacks: list[callable], chat_model_used: bool
):
    response_text: str = ""
    for chunk in response:
        if text_chunk := _get_chunk_text(chunk, chat_model_used):
            response_text += text_chunk
            [cb(text_chunk) for cb in callbacks]
    return LLMResponse(response_text, {})


def _prepare_llm_arguments(config: Config, kwargs: dict):
    args = {**config.LLM_DEFAULT_ARGS, **kwargs}
    args["model"] = args.get(
        "model",
        args.get("deployment_id", config.LLM_DEPLOYMENT_ID or config.MODEL)
        if config.LLM_API_TYPE == ApiType.AZURE
        else config.MODEL,
    )
    callbacks: list[callable] = args.pop("callbacks", [])
    if "callback" in args:
        cb = args.pop("callback")
        if cb:
            callbacks.append(cb)
    args["stream"] = bool(callbacks)
    return args, {"callbacks": callbacks}


def make_llm_functions(config: Config) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    # _configure_open_ai_package(config)

    if config.LLM_API_TYPE == ApiType.AZURE:
        connection_type = openai.AzureOpenAI
        async_connection_type = openai.AsyncAzureOpenAI
        params = dict(
            api_key=config.LLM_API_KEY,
            azure_endpoint=config.LLM_API_BASE,
            api_version=config.LLM_API_VERSION,
        )
    else:
        connection_type = openai.OpenAI
        async_connection_type = openai.AsyncOpenAI
        params = dict(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE,
        )

    _connection = connection_type(**params)
    _async_connection = async_connection_type(**params)

    async def allm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        if is_chat_model(args["model"]):
            # _connection.chat.completions.create()

            response = await _async_connection.chat.completions.create(
                messages=prepare_chat_messages(prompt), **args
            )
            if args["stream"]:
                return await _a_process_streamed_response(
                    response, options["callbacks"], chat_model_used=True
                )

            for cb in options["callbacks"]:
                cb(response.choices[0].message.content)
            return LLMResponse(response.choices[0].message.content, response.__dict__)

        response = await _async_connection.completions.create(
            prompt=prepare_prompt(prompt), **args
        )
        if args["stream"]:
            return await _a_process_streamed_response(
                response, options["callbacks"], chat_model_used=False
            )

        return LLMResponse(response.choices[0].text, response.__dict__)

    def llm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        if is_chat_model(args["model"]):
            response = _connection.chat.completions.create(
                messages=prepare_chat_messages(prompt), **args
            )
            if args["stream"]:
                return _process_streamed_response(
                    response, options["callbacks"], chat_model_used=True
                )

            for cb in options["callbacks"]:
                cb(response.choices[0].message.content)
            return LLMResponse(response.choices[0].message.content, response.__dict__)

        # Else (if it is text completion model)
        response = _connection.completions.create(prompt=prepare_prompt(prompt), **args)
        if args["stream"]:
            return _process_streamed_response(
                response, options["callbacks"], chat_model_used=False
            )

        return LLMResponse(response.choices[0].text, response.__dict__)

    return llm, allm
