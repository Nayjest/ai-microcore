import asyncio
import base64

import openai
from openai.types import CompletionChoice, ImagesResponse

from ..lm_client import BaseAIChatClient, BaseAsyncAIClient
from ..message_types import TMsgContentPart
from ..configuration import Config
from ..llm_backends import ApiPlatform
from .._prepare_llm_args import prepare_prompt
from ..types import BadAIAnswer, TPrompt
from ..wrappers.llm_response_wrapper import (
    LLMResponse,
    ImageGenerationResponse,
    StoredImageGenerationResponse
)
from ..utils import is_chat_model, is_image_model
from .shared import make_image_generation_response, make_remove_hidden_output, prepare_callbacks
from ..images import (
    Image,
    FileImage,
    ImageInterface,
    ImageListInterface,
    image_format_to_mime_type
)


class AsyncOpenAIClient(BaseAsyncAIClient):
    oai_client: openai.AsyncOpenAI | openai.AsyncAzureOpenAI
    sync_client: "OpenAIClient"

    def __init__(self, oai_connection, sync_client: "BaseAIChatClient"):
        self.sync_client = sync_client
        self.oai_client = oai_connection

    async def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse | ImageGenerationResponse | StoredImageGenerationResponse:
        config = self.sync_client.config
        args, options = _prepare_llm_arguments(config, kwargs)
        if is_image_model(args["model"]):
            return await _generate_image_async(
                prompt,
                args,
                self.oai_client,
                options
            )
        if is_chat_model(args["model"], config):
            messages = self.sync_client.convert_prompt_to_chat_input(prompt)
            response = await self.oai_client.chat.completions.create(
                messages=messages, **args
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
                response_text = self.sync_client.remove_hidden_output(response_text)
            for cb in options["callbacks"]:
                if asyncio.iscoroutinefunction(cb):
                    await cb(response_text)
                else:
                    cb(response_text)
            return LLMResponse(response_text, response.__dict__)

        response = await self.oai_client.completions.create(
            prompt=prepare_prompt(prompt), **args
        )
        check_for_errors(response)
        if args["stream"]:
            return await _a_process_streamed_response(
                response, options["callbacks"], chat_model_used=False
            )
        return LLMResponse(response.choices[0].text, response.__dict__)

    async def load_models(self) -> dict:
        models_iter = self.oai_client.models.list()
        return {model.id: model async for model in models_iter}


class OpenAIClient(BaseAIChatClient):
    aio: "AsyncOpenAIClient"
    oai_client: openai.OpenAI | openai.AzureOpenAI
    remove_hidden_output: callable

    def __init__(self, config: Config):
        super().__init__(config)
        if config.LLM_API_PLATFORM == ApiPlatform.AZURE:
            client_type = openai.AzureOpenAI
            async_client_type = openai.AsyncAzureOpenAI
            client_params = {
                "api_key": config.LLM_API_KEY,
                "azure_endpoint": config.LLM_API_BASE,
                "api_version": config.LLM_API_VERSION,
                **config.INIT_PARAMS,
            }
        else:
            client_type = openai.OpenAI
            async_client_type = openai.AsyncOpenAI
            client_params = {
                "api_key": config.LLM_API_KEY,
                "base_url": config.LLM_API_BASE,
                **config.INIT_PARAMS,
            }
        self.oai_client = client_type(**client_params)
        self.aio = AsyncOpenAIClient(
            oai_connection=async_client_type(**client_params),
            sync_client=self
        )
        self.remove_hidden_output: callable = make_remove_hidden_output(config)

    def _convert_message_content_part(
        self,
        content_part: TMsgContentPart,
        converted_content: list = None
    ) -> dict | list[dict] | None:
        """
        Convert the message content part into a format suitable for the LLM inference chat API.
        """
        if isinstance(content_part, str):
            return {"type": "text", "text": content_part}
        if isinstance(img := content_part, ImageInterface):
            return image_to_oai(img)
        return content_part

    def load_models(self, **kwargs) -> dict:
        models_iter = self.oai_client.models.list(**kwargs)
        return {model.id: model for model in models_iter}

    def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse | ImageGenerationResponse | StoredImageGenerationResponse:
        args, options = _prepare_llm_arguments(self.config, kwargs)
        if is_image_model(args["model"]):
            return _generate_image(
                prompt,
                args,
                self.oai_client,
                options
            )
        is_chat: bool = is_chat_model(args["model"], self.config)
        if is_chat:
            messages = self.convert_prompt_to_chat_input(prompt)
            response = self.oai_client.chat.completions.create(
                messages=messages, **args
            )
        else:
            response = self.oai_client.completions.create(prompt=prompt, **args)

        check_for_errors(response)
        if args["stream"]:
            return _process_streamed_response(
                response,
                options["callbacks"],
                chat_model_used=is_chat,
                hidden_output_begin=self.config.HIDDEN_OUTPUT_BEGIN,
                hidden_output_end=self.config.HIDDEN_OUTPUT_END,
            )
        choice = response.choices[0]
        if is_chat and not isinstance(choice, CompletionChoice):
            response_text = choice.message.content
        else:
            response_text = choice.text

        if self.config.hiding_output():
            response_text = self.remove_hidden_output(response_text)
        for cb in options["callbacks"]:
            cb(response_text)
        return LLMResponse(response_text, response.__dict__)


def image_to_oai(img: ImageInterface) -> dict:
    b64_data = base64.b64encode(img.get_bytes()).decode()
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{img.mime_type()};base64,{b64_data}"
        }
    }


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
            if config.LLM_API_PLATFORM == ApiPlatform.AZURE
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


def _oai_image_response_to_images(response: ImagesResponse) -> list[Image]:
    images = []
    for oai_img in response.data:
        image_bytes = base64.b64decode(oai_img.b64_json)
        img = Image(
            image_bytes,
            mime_type=image_format_to_mime_type(str(response.output_format))
        )
        images.append(img)
    return images


def _prepare_image_generation(prompt, args):
    """Prepare prompt and images for image generation (shared logic)."""
    def convert_input_image(image: ImageInterface):
        if isinstance(image, FileImage):
            return open(image.file, "rb")
        return image.get_bytes()

    images = []
    if isinstance(prompt, list):
        items = prompt
        prompt = ""
        for item in items:
            if isinstance(item, dict) and "content" in item:
                if isinstance(item, ImageListInterface):
                    for img in item.images():
                        images.append(convert_input_image(img))
                elif isinstance(item, ImageInterface):
                    images.append(convert_input_image(item))
                else:
                    prompt += str(item["content"])
            elif isinstance(item, ImageListInterface):
                for img in item.images():
                    images.append(convert_input_image(img))
            elif isinstance(item, ImageInterface):
                images.append(convert_input_image(item))
            else:
                prompt += str(item)

    save: bool = args.pop("save", True)
    args.pop("stream", None)
    if args["model"] in ["dall-e-2", "dall-e-3"] and "response_format" not in args:
        args["response_format"] = "b64_json"
    if save and args.get("response_format", "b64_json") != "b64_json":
        raise ValueError("Only 'b64_json' response format is supported.")

    return prompt, images, save


def _image_generation_response(
        response: ImagesResponse,
        save: bool,
        options: dict,
) -> ImageGenerationResponse | None:
    check_for_errors(response)
    images = _oai_image_response_to_images(response)
    response_attrs = response.__dict__.copy()
    result = make_image_generation_response(images, save, response_attrs)
    for cb in options["callbacks"]:
        cb(result)
    return result


def _generate_image(
    prompt,
    args,
    connection: openai.OpenAI,
    options
) -> ImageGenerationResponse | None:
    """Synchronous version of image generation."""
    prompt, images, save = _prepare_image_generation(prompt, args)

    if not images:
        response: ImagesResponse = connection.images.generate(prompt=prompt, **args)
    else:
        response: ImagesResponse = connection.images.edit(
            image=images,
            prompt=prompt,
            **args
        )
    return _image_generation_response(response, save, options)


async def _generate_image_async(
    prompt,
    args,
    connection: openai.AsyncOpenAI,
    options
) -> ImageGenerationResponse | None:
    """Asynchronous version of image generation."""
    prompt, images, save = _prepare_image_generation(prompt, args)

    if not images:
        response: ImagesResponse = await connection.images.generate(prompt=prompt, **args)
    else:
        response: ImagesResponse = await connection.images.edit(
            image=images,
            prompt=prompt,
            **args
        )
    return _image_generation_response(response, save, options)
