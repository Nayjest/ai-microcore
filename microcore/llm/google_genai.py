import asyncio
from google.ai.generativelanguage import Content, Part
from google.generativeai import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from ..configuration import Config
from .._prepare_llm_args import prepare_chat_messages
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


def make_llm_functions(config: Config) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    genai.configure(api_key=config.LLM_API_KEY, **config.INIT_PARAMS)
    if config.GOOGLE_GEMINI_SAFETY_SETTINGS is None:
        # Only new categories
        config.GOOGLE_GEMINI_SAFETY_SETTINGS = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _prepare_chat(prompt, **kwargs):
        model_name = kwargs.pop("model", config.MODEL)
        callbacks = prepare_callbacks(config, kwargs, set_stream=False)
        model = genai.GenerativeModel(
            model_name,
            generation_config=GenerationConfig(**kwargs),
            safety_settings=config.GOOGLE_GEMINI_SAFETY_SETTINGS,
        )
        messages = _chat_messages_to_google(prepare_chat_messages(prompt))
        last_message = messages.pop()
        chat = model.start_chat(history=messages)
        return chat, last_message.parts[0], callbacks

    async def allm(prompt, **kwargs):
        chat, msg, callbacks = _prepare_chat(prompt, **kwargs)
        try:
            response = await chat.send_message_async(msg, stream=bool(callbacks))
            if callbacks:
                return await _a_process_streamed_response(response, callbacks)
            return LLMResponse(response.text, response.__dict__)
        except ValueError as e:
            raise BadAIAnswer(str(e)) from e

    def llm(prompt, **kwargs):
        chat, msg, callbacks = _prepare_chat(prompt, **kwargs)
        try:
            response = chat.send_message(msg, stream=bool(callbacks))
            if callbacks:
                return _process_streamed_response(response, callbacks)
            return LLMResponse(response.text, response.__dict__)
        except ValueError as e:
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
        Content(role=msg["role"], parts=[Part(text=msg["content"])])
        for msg in merged_msg
    ]
    return vertex_messages
