import asyncio
import base64
import logging

import openai

from ..configuration import Config, ApiType
from .._prepare_llm_args import prepare_chat_messages, prepare_prompt
from ..types import LLMAsyncFunctionType, LLMFunctionType, BadAIAnswer
from ..wrappers.llm_response_wrapper import LLMResponse
from ..utils import file_link, is_chat_model, is_image_model
from .shared import make_remove_hidden_output, prepare_callbacks
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
    response,
    callbacks: list[callable],
    chat_model_used: bool,
    hidden_output_begin: str | None = None,
    hidden_output_end: str | None = None,
):
    response_text: str = ""
    hiding: bool = False
    need_to_hide = hidden_output_begin and hidden_output_end
    async for chunk in response:
        if text_chunk := _get_chunk_text(chunk, chat_model_used):
            if need_to_hide:
                if text_chunk == hidden_output_begin:
                    hiding = True
                    continue
                else:
                    if hiding:
                        if text_chunk == hidden_output_end:
                            hiding = False
                            text_chunk = ""
                        else:
                            continue
            response_text += text_chunk
            for cb in callbacks:
                if asyncio.iscoroutinefunction(cb):
                    await cb(text_chunk)
                else:
                    cb(text_chunk)
    return LLMResponse(response_text, {})


def _process_streamed_response(
    response,
    callbacks: list[callable],
    chat_model_used: bool,
    hidden_output_begin: str | None = None,
    hidden_output_end: str | None = None,
):
    response_text: str = ""
    is_hiding: bool = False
    need_to_hide = hidden_output_begin and hidden_output_end
    for chunk in response:
        if text_chunk := _get_chunk_text(chunk, chat_model_used):
            if need_to_hide:
                if text_chunk == hidden_output_begin:
                    is_hiding = True
                    continue
                else:
                    if is_hiding:
                        if text_chunk == hidden_output_end:
                            is_hiding = False
                            text_chunk = ""
                        else:
                            continue
            response_text += text_chunk
            [cb(text_chunk) for cb in callbacks]
    return LLMResponse(response_text, {})


def _prepare_llm_arguments(config: Config, kwargs: dict):
    args = {**config.LLM_DEFAULT_ARGS, **kwargs}
    args["model"] = args.get(
        "model",
        (
            args.get("deployment_id", config.LLM_DEPLOYMENT_ID or config.MODEL)
            if config.LLM_API_TYPE == ApiType.AZURE
            else config.MODEL
        ),
    )
    callbacks = prepare_callbacks(config, args)
    return args, {"callbacks": callbacks}


def check_for_errors(response):
    if hasattr(response, "object") and response.object == "error":
        raise BadAIAnswer(response.message)
    if hasattr(response, "error") and response.error:
        raise BadAIAnswer(str(response.error))


def make_llm_functions(config: Config) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    if config.LLM_API_TYPE == ApiType.AZURE:
        connection_type = openai.AzureOpenAI
        async_connection_type = openai.AsyncAzureOpenAI
        params = {
            "api_key": config.LLM_API_KEY,
            "azure_endpoint": config.LLM_API_BASE,
            "api_version": config.LLM_API_VERSION,
            **config.INIT_PARAMS,
        }
    else:
        connection_type = openai.OpenAI
        async_connection_type = openai.AsyncOpenAI
        params = {
            "api_key": config.LLM_API_KEY,
            "base_url": config.LLM_API_BASE,
            **config.INIT_PARAMS,
        }

    _connection = connection_type(**params)
    _async_connection = async_connection_type(**params)
    remove_hidden_output: callable = make_remove_hidden_output(config)

    async def allm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        if is_chat_model(args["model"], config):
            response = await _async_connection.chat.completions.create(
                messages=prepare_chat_messages(prompt), **args
            )
            check_for_errors(response)
            if args["stream"]:
                return await _a_process_streamed_response(
                    response,
                    options["callbacks"],
                    chat_model_used=True,
                    hidden_output_begin=config.HIDDEN_OUTPUT_BEGIN,
                    hidden_output_end=config.HIDDEN_OUTPUT_END,
                )
            response_text = response.choices[0].message.content
            if config.hiding_output():
                response_text = remove_hidden_output(response_text)
            for cb in options["callbacks"]:
                if asyncio.iscoroutinefunction(cb):
                    await cb(response_text)
                else:
                    cb(response_text)
            return LLMResponse(response_text, response.__dict__)

        response = await _async_connection.completions.create(
            prompt=prepare_prompt(prompt), **args
        )
        check_for_errors(response)
        if args["stream"]:
            return await _a_process_streamed_response(
                response, options["callbacks"], chat_model_used=False
            )
        return LLMResponse(response.choices[0].text, response.__dict__)

    def llm(prompt, **kwargs):
        args, options = _prepare_llm_arguments(config, kwargs)
        if is_image_model(args["model"]):
            save: bool = args.pop("save", True)
            args.pop("stream", None)
            if args["model"] in ["dall-e-2", "dall-e-3"] and "response_format" not in args:
                args["response_format"] = "b64_json"
            if save and args.get("response_format", "b64_json") != "b64_json":
                raise ValueError("Can only save images with response_format='b64_json'")
            response = _connection.images.generate(prompt=prompt, **args)
            check_for_errors(response)
            response_attrs = response.__dict__.copy()
            img_repr = "<image>"
            if save:
                from ..file_storage import storage
                file_name = save if isinstance(save, str) else "generated_images/image.png"
                response_attrs["files"] = []
                for i, img in enumerate(response.data):
                    image_bytes = base64.b64decode(img.b64_json)
                    actual_fn = storage.write(file_name, image_bytes, rewrite_existing=False)
                    actual_fn = storage.abs_path(actual_fn)
                    logging.info(f"Image saved to {file_link(actual_fn)}")
                    response_attrs["files"].append(actual_fn)
                response_attrs["file"] = (
                    response_attrs["files"][0] if response_attrs["files"]
                    else None
                )
                if len(response_attrs["files"]) == 1:
                    img_repr = file_link(response_attrs['file'])
                elif len(response_attrs["files"]) > 1:
                    img_repr = "\n".join(response_attrs["files"])
            for cb in options["callbacks"]:
                cb(img_repr)
            return LLMResponse(img_repr, response_attrs)
        if is_chat_model(args["model"], config):
            response = _connection.chat.completions.create(
                messages=prepare_chat_messages(prompt), **args
            )
            check_for_errors(response)
            if args["stream"]:
                return _process_streamed_response(
                    response,
                    options["callbacks"],
                    chat_model_used=True,
                    hidden_output_begin=config.HIDDEN_OUTPUT_BEGIN,
                    hidden_output_end=config.HIDDEN_OUTPUT_END,
                )
            response_text = response.choices[0].message.content

            if config.hiding_output():
                response_text = remove_hidden_output(response_text)
            for cb in options["callbacks"]:
                cb(response_text)
            return LLMResponse(response_text, response.__dict__)

        # Else (if it is text completion model)
        response = _connection.completions.create(prompt=prepare_prompt(prompt), **args)
        check_for_errors(response)
        if args["stream"]:
            return _process_streamed_response(
                response, options["callbacks"], chat_model_used=False
            )
        return LLMResponse(response.choices[0].text, response.__dict__)

    return llm, allm
