import asyncio

from ..config import Config, ApiType
from ..extended_string import ExtendedString
from ..prepare_llm_args import prepare_chat_messages, prepare_prompt
from ..types import LLMAsyncFunctionType, LLMFunctionType
from ..utils import is_chat_model
import openai
import openai.util


def _get_chunk_text(chunk, mode_chat_model: bool):
    # Azure API gives first chunk with empty choices
    choice = chunk.choices[0] if len(chunk.choices) else {}

    if mode_chat_model:
        return choice.get("delta", {}).get("content", "")
    else:
        return choice.get("text", "")


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
    return ExtendedString(response_text, {})


def _process_streamed_response(
        response, callbacks: list[callable], chat_model_used: bool
):
    response_text: str = ""
    for chunk in response:
        if text_chunk := _get_chunk_text(chunk, chat_model_used):
            response_text += text_chunk
            [cb(text_chunk) for cb in callbacks]
    return ExtendedString(response_text, {})


def _configure_open_ai_package(config: Config):
    try:
        api_type = config.LLM_API_TYPE
        openai.util.ApiType.from_str(api_type)
    except openai.error.InvalidAPIType:
        api_type = ApiType.OPEN_AI

    openai.api_type = api_type
    openai.api_key = config.LLM_API_KEY
    openai.api_base = config.LLM_API_BASE
    openai.api_version = config.LLM_API_VERSION


def _prepare_llm_arguments(config: Config, kwargs: dict):
    args = {**config.LLM_DEFAULT_ARGS, **kwargs}

    args["model"] = args.get("model", config.MODEL)

    if config.LLM_API_TYPE == ApiType.AZURE:
        args["deployment_id"] = args.get("deployment_id", config.LLM_DEPLOYMENT_ID)

    callbacks: list[callable] = args.pop("callbacks", [])
    if "callback" in args:
        callbacks.append(args.pop("callback"))
    args["stream"] = bool(callbacks)
    return args, dict(callbacks=callbacks)


def make_llm_functions(config: Config) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    _configure_open_ai_package(config)

    async def allm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        if is_chat_model(args["model"]):
            response = await openai.ChatCompletion.acreate(
                messages=prepare_chat_messages(prompt), **args
            )
            if args["stream"]:
                return await _a_process_streamed_response(
                    response, options['callbacks'], chat_model_used=True
                )
            else:
                for cb in options['callbacks']:
                    cb(response.choices[0].message.content)
                return ExtendedString(response.choices[0].message.content, response)
        else:
            response = await openai.Completion.acreate(
                prompt=prepare_prompt(prompt), **args
            )
            if args["stream"]:
                return await _a_process_streamed_response(
                    response, options['callbacks'], chat_model_used=False
                )
            else:
                return ExtendedString(response.choices[0].text, response)

    def llm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        if is_chat_model(args["model"]):
            response = openai.ChatCompletion.create(
                messages=prepare_chat_messages(prompt), **args
            )
            if args["stream"]:
                return _process_streamed_response(
                    response, options['callbacks'], chat_model_used=True
                )
            else:
                for cb in options['callbacks']:
                    cb(response.choices[0].message.content)
                return ExtendedString(response.choices[0].message.content, response)
        else:
            response = openai.Completion.create(
                prompt=prepare_prompt(prompt), **args
            )
            if args["stream"]:
                return _process_streamed_response(
                    response, options['callbacks'], chat_model_used=False
                )
            else:
                return ExtendedString(response.choices[0].text, response)

    return llm, allm
