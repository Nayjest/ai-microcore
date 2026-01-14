"""
Google Vertex AI LLM functions.
@deprecated Use Google GenAI instead.
"""
import asyncio
import logging
import os
from vertexai.generative_models import (
    Content,
    Part,
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
    ResponseValidationError,
)
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.oauth2.credentials import Credentials

from ..configuration import Config
from .._prepare_llm_args import prompt_to_message_dicts
from ..message_types import Role
from ..types import LLMAsyncFunctionType, LLMFunctionType, BadAIAnswer
from ..wrappers.llm_response_wrapper import LLMResponse
from .shared import prepare_callbacks


async def _a_process_streamed_response(response, callbacks: list[callable]):
    response_text: str = ""
    async for chunk in response:
        if text_chunk := chunk.text:
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
        if text_chunk := chunk.text:
            response_text += text_chunk
            [cb(text_chunk) for cb in callbacks]
    return LLMResponse(response_text, {})


def init_vertex_ai(config: Config):
    if (not config.GOOGLE_VERTEX_ACCESS_TOKEN) and config.GOOGLE_VERTEX_GCLOUD_AUTH:
        logging.info("Authenticating with Google Cloud...")
        config.GOOGLE_VERTEX_ACCESS_TOKEN = (
            os.popen("gcloud auth application-default print-access-token")
            .read()
            .replace("Python", "")
            .strip()
        )
        if not config.GOOGLE_VERTEX_ACCESS_TOKEN:
            raise ValueError(
                "Failed to authenticate with Google Cloud. "
                "Please make sure you have gcloud installed and configured "
                "(try `gcloud auth application-default login`; "
                "`gcloud auth application-default print-access-token`)."
            )
    credentials = Credentials(token=config.GOOGLE_VERTEX_ACCESS_TOKEN)
    defaults = dict(
        credentials=credentials,
        project=config.GOOGLE_VERTEX_PROJECT_ID,
        location=config.GOOGLE_VERTEX_LOCATION or None,
        api_endpoint=config.LLM_API_BASE or None,
    )
    vertexai.init(**{**defaults, **config.INIT_PARAMS})


def make_llm_functions(config: Config) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    init_vertex_ai(config)
    if config.GOOGLE_GEMINI_SAFETY_SETTINGS is None:
        config.GOOGLE_GEMINI_SAFETY_SETTINGS = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _prepare_chat(prompt, **kwargs):
        model_name = kwargs.pop("model", config.MODEL)
        callbacks = prepare_callbacks(config, kwargs, set_stream=False)
        model = GenerativeModel(
            model_name,
            generation_config=GenerationConfig(**kwargs),
            safety_settings=config.GOOGLE_GEMINI_SAFETY_SETTINGS,
        )
        messages = _chat_messages_to_google(prompt_to_message_dicts(prompt))
        last_message = messages.pop()
        chat = model.start_chat(
            history=messages,
            response_validation=config.GOOGLE_VERTEX_RESPONSE_VALIDATION,
        )
        return chat, last_message.parts[0], callbacks

    async def allm(prompt, **kwargs):
        chat, msg, callbacks = _prepare_chat(prompt, **kwargs)
        try:
            response = await chat.send_message_async(msg, stream=bool(callbacks))
            if callbacks:
                return await _a_process_streamed_response(response, callbacks)
            return LLMResponse(response.text, response.__dict__)
        except (ResponseValidationError, ValueError) as e:
            raise BadAIAnswer(str(e)) from e

    def llm(prompt, **kwargs):
        chat, msg, callbacks = _prepare_chat(prompt, **kwargs)
        try:
            response = chat.send_message(msg, stream=bool(callbacks))
            if callbacks:
                return _process_streamed_response(response, callbacks)
            return LLMResponse(response.text, response.__dict__)
        except (ResponseValidationError, ValueError) as e:
            raise BadAIAnswer(str(e)) from e

    return llm, allm


def _chat_messages_to_google(messages: list[dict]):
    # Convert roles to Google Vertex roles
    # (system,user,assistant) -> (user, model)
    for msg in messages:
        if msg["role"] == Role.SYSTEM:
            msg["role"] = "user"
        elif msg["role"] == Role.ASSISTANT:
            msg["role"] = "model"

    # Merge sequences of messages with same role
    # to avoid following error:
    #   google.api_core.exceptions.InvalidArgument:
    #   400 Please ensure that multiturn requests ends with a user role or a function response.
    merged_msg = []
    for msg in messages:
        if merged_msg and msg["role"] == merged_msg[-1]["role"]:
            merged_msg[-1]["content"] += "\n" + msg["content"]
        else:
            merged_msg.append(msg)

    vertex_messages = [
        Content(role=msg["role"], parts=[Part.from_text(msg["content"])])
        for msg in merged_msg
    ]
    return vertex_messages
