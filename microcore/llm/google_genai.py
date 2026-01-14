import asyncio
from typing import Any
from dataclasses import dataclass

from google import genai
from google.genai import types

from ..configuration import Config
from ..message_types import Role, TMsgContentPart
from ..types import BadAIAnswer, TPrompt
from ..wrappers.llm_response_wrapper import (
    LLMResponse,
    ImageGenerationResponse,
    StoredImageGenerationResponse
)
from ..utils import is_image_model
from .shared import make_image_generation_response, prepare_callbacks
from ..lm_client import BaseAsyncAIClient, BaseAIChatClient
from ..images import Image, ImageInterface


class GoogleClient(BaseAIChatClient):
    aio: "AsyncGoogleClient"

    def __init__(self, config: Config):
        super().__init__(config)
        self.genai_client = genai.Client(api_key=config.LLM_API_KEY, **config.INIT_PARAMS)
        self.aio = AsyncGoogleClient(self)

    def load_models(self, **kwargs) -> dict:
        models_iter = self.genai_client.models.list(**kwargs)
        return {model.name: model for model in models_iter}

    def convert_prompt_to_chat_input(self, prompt: TPrompt) -> list[dict | Any]:
        messages = super().convert_prompt_to_chat_input(prompt)
        return _convert_message_roles(messages)

    def _convert_message(self, message: dict) -> dict | Any:
        message["parts"] = self._convert_message_content(message.pop("content"))
        return message

    def _convert_message_content_part(
        self,
        content_part: TMsgContentPart,
        converted_content: list = None
    ) -> types.Part | list[types.Part] | None:
        """
        Convert the message content part into a format suitable for the LLM inference chat API.
        """
        if isinstance(content_part, types.Part):
            return content_part
        if isinstance(img := content_part, ImageInterface):
            return types.Part.from_bytes(data=img.get_bytes(), mime_type=img.mime_type())
        if isinstance(content_part, str):
            return types.Part(text=content_part)
        return content_part

    def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse | ImageGenerationResponse | StoredImageGenerationResponse:
        ctx = _GenerationContext.create(self, prompt, kwargs)
        try:
            if not ctx.stream:
                response: genai.types.GenerateContentResponse = \
                    ctx.genai_client.models.generate_content(
                        model=ctx.model_name,
                        contents=ctx.messages,
                        config=ctx.gen_config
                    )
                if ctx.is_image_model:
                    images = _google_image_response_to_images(response)
                    return make_image_generation_response(images, ctx.save, response.__dict__)
                return LLMResponse(response.text, response.__dict__)
            else:
                response_iterator = ctx.genai_client.models.generate_content_stream(
                    model=ctx.model_name,
                    contents=ctx.messages,
                    config=ctx.gen_config
                )
                return _process_streamed_response(response_iterator, ctx.callbacks)
        except ValueError as e:
            raise BadAIAnswer(str(e)) from e


class AsyncGoogleClient(BaseAsyncAIClient):

    def __init__(self, client: GoogleClient):
        self.sync_client = client

    async def load_models(self) -> dict:
        raise NotImplementedError

    async def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse | ImageGenerationResponse | StoredImageGenerationResponse:
        ctx = _GenerationContext.create(self.sync_client, prompt, kwargs)
        try:
            if not ctx.stream:
                response: genai.types.GenerateContentResponse = \
                    await ctx.genai_client.aio.models.generate_content(
                        model=ctx.model_name,
                        contents=ctx.messages,
                        config=ctx.gen_config
                    )
                if ctx.is_image_model:
                    images = _google_image_response_to_images(response)
                    return make_image_generation_response(images, ctx.save, response.__dict__)
                return LLMResponse(response.text, response.__dict__)
            else:
                response_iterator = await ctx.genai_client.aio.models.generate_content_stream(
                    model=ctx.model_name,
                    contents=ctx.messages,
                    config=ctx.gen_config
                )
                return await _a_process_streamed_response(response_iterator, ctx.callbacks)
        except ValueError as e:
            raise BadAIAnswer(str(e)) from e


@dataclass
class _GenerationContext:
    """
    Context for a single generation request.

    It's organized this way to avoid sync / async code duplication.
    """
    genai_client: genai.Client
    config: Config
    model_name: str
    save: bool
    messages: list[dict]
    callbacks: list[callable]
    gen_config: genai.types.GenerateContentConfig
    is_image_model: bool
    stream: bool = False

    @staticmethod
    def create(
        client: "GoogleClient",
        prompt: TPrompt,
        kwargs: dict
    ) -> "_GenerationContext":
        model_name = kwargs.pop("model", client.config.MODEL)
        callbacks = prepare_callbacks(client.config, kwargs, set_stream=False)
        is_image = is_image_model(model_name)
        return _GenerationContext(
            model_name=model_name,
            save=kwargs.pop("save", True),
            messages=client.convert_prompt_to_chat_input(prompt),
            callbacks=callbacks,
            gen_config=genai.types.GenerateContentConfig(**kwargs),
            genai_client=client.genai_client,
            config=client.config,
            is_image_model=is_image,
            stream=callbacks and not is_image
        )


def _google_image_response_to_images(response: genai.types.GenerateContentResponse) -> list[Image]:
    images = []
    for i, part in enumerate(response.parts):
        if part.inline_data:
            image_bytes = part.inline_data.data
            img = Image(image_bytes, mime_type=part.inline_data.mime_type)
            images.append(img)
    return images


def _convert_message_roles(messages: list[dict]):
    """
    Convert roles to Google Vertex roles
    (system,user,assistant) -> (user, model).
    """
    for msg in messages:
        if msg["role"] == Role.SYSTEM:
            msg["role"] = "user"
        elif msg["role"] == Role.ASSISTANT:
            msg["role"] = "model"
    return messages


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
