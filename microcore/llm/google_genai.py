import asyncio
import json
from typing import Any, Optional
from dataclasses import dataclass

from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials

from ..configuration import Config, LLMCredentialError
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


def _load_service_account_info(config: Config) -> Optional[dict]:
    """
    Load Google Cloud service account information from a file or JSON string.
    Returns the service account info as a dictionary.
    Raises LLMCredentialError if loading fails.
    """
    if config.GOOGLE_CLOUD_SERVICE_ACCOUNT:
        if isinstance(config.GOOGLE_CLOUD_SERVICE_ACCOUNT, dict):
            return config.GOOGLE_CLOUD_SERVICE_ACCOUNT
        try:
            with open(config.GOOGLE_CLOUD_SERVICE_ACCOUNT, 'r') as f:
                creds_info = json.load(f)
                return creds_info
        except FileNotFoundError:
            raise LLMCredentialError(
                f"Service account file not found: {config.GOOGLE_CLOUD_SERVICE_ACCOUNT}"
            )
        except PermissionError:
            raise LLMCredentialError(
                f"Permission denied reading service account file: "
                f"{config.GOOGLE_CLOUD_SERVICE_ACCOUNT}"
            )
        except json.JSONDecodeError as e:
            raise LLMCredentialError(
                f"Invalid JSON in service account file {config.GOOGLE_CLOUD_SERVICE_ACCOUNT}: {e}"
            )
    elif config.GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON:
        if isinstance(config.GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON, dict):
            return config.GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON
        try:
            creds_info = json.loads(config.GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON)
            return creds_info
        except json.JSONDecodeError as e:
            raise LLMCredentialError(
                f"Invalid JSON in GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON: {e}"
            )
    return None


class GoogleClient(BaseAIChatClient):
    """
    Client for Google GenAI SDK to interact with Google Gemini models.
    """
    genai_client: genai.Client
    aio: "AsyncGoogleClient"

    def __init__(self, config: Config):
        super().__init__(config)
        client_params = {**config.INIT_PARAMS}

        if config.GOOGLE_GENAI_USE_VERTEXAI is not None:
            client_params["vertexai"] = config.GOOGLE_GENAI_USE_VERTEXAI

        if config.GOOGLE_CLOUD_PROJECT_ID:
            client_params["project"] = config.GOOGLE_CLOUD_PROJECT_ID

        if config.GOOGLE_CLOUD_LOCATION:
            client_params["location"] = config.GOOGLE_CLOUD_LOCATION

        if creds_info := _load_service_account_info(config):
            try:
                credentials = Credentials.from_service_account_info(
                    creds_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except ValueError as e:
                raise LLMCredentialError(
                    f"Invalid service account info provided: {e}"
                ) from e
            client_params["credentials"] = credentials
            if "project" not in client_params and creds_info.get("project_id"):
                client_params["project"] = creds_info["project_id"]
            if "vertexai" not in client_params:
                client_params["vertexai"] = True
        else:
            client_params["api_key"] = config.LLM_API_KEY

        self.genai_client = genai.Client(**client_params)
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

    async def load_models(self, **kwargs) -> dict:
        models = await self.sync_client.genai_client.aio.models.list(**kwargs)
        return {model.name: model for model in models}

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
        stream = kwargs.pop("stream", False) or (callbacks and not is_image)
        return _GenerationContext(
            model_name=model_name,
            save=kwargs.pop("save", True),
            messages=client.convert_prompt_to_chat_input(prompt),
            callbacks=callbacks,
            gen_config=genai.types.GenerateContentConfig(**kwargs),
            genai_client=client.genai_client,
            config=client.config,
            is_image_model=is_image,
            stream=stream,
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
