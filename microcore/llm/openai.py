"""
OpenAI LLM client implementation.
"""
import asyncio
import base64
from typing import Any, Awaitable, Callable
import inspect

import openai
from openai.types import CompletionChoice, ImagesResponse

from ..lm_client import BaseAIChatClient, BaseAsyncAIClient
from ..message_types import TMsgContentPart, TMsgContent
from ..configuration import Config, LLMConfigError
from ..llm_backends import ApiType, ApiPlatform
from .._prepare_llm_args import prepare_prompt
from ..types import BadAIAnswer, TPrompt
from ..wrappers.llm_response_wrapper import (
    LLMResponse,
    ImageGenerationResponse,
    StoredImageGenerationResponse
)
from ..utils import is_chat_model, is_image_model
from .azure_responses import (
    adapt_responses_events,
    adapt_responses_events_async,
    build_responses_client_params,
    build_responses_request,
    extract_responses_text,
    should_use_responses_api,
)
from .shared import (
    attrs_with_normalized_usage,
    ensure_stream_include_usage,
    make_image_generation_response,
    make_remove_hidden_output,
    prepare_callbacks,
    streaming_usage_attrs,
)
from ..images import (
    Image,
    FileImage,
    ImageInterface,
    ImageListInterface,
    image_format_to_mime_type
)


def _as_async_str_provider(
    provider: Callable[[], Any],
) -> Callable[[], Awaitable[str]]:
    """Make a sync token provider awaitable for openai.AsyncOpenAI (SDK 1.109+).

    Does not move azure.identity off the event loop: ``provider()`` still blocks,
    same as before. This only avoids ``await str`` TypeError.
    """
    if asyncio.iscoroutinefunction(provider):
        return provider

    async def _provide() -> str:
        value = provider()
        if inspect.isawaitable(value):
            value = await value
        return value

    return _provide


def _async_openai_client_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    for key in ("api_key", "azure_ad_token_provider"):
        value = out.get(key)
        if callable(value):
            out[key] = _as_async_str_provider(value)
    return out


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
        if should_use_responses_api(config, options["use_responses_api"]):
            _ensure_responses_available(self.sync_client.responses_api_available)
            return await _generate_via_responses_async(
                self,
                prompt,
                args,
                options,
                config,
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
            response_text: str = response.choices[0].message.content or ""
            if config.hiding_output():
                response_text = self.sync_client.remove_hidden_output(response_text)
            for cb in options["callbacks"]:
                if asyncio.iscoroutinefunction(cb):
                    await cb(response_text)
                else:
                    cb(response_text)
            return LLMResponse(
                response_text,
                attrs_with_normalized_usage(response.__dict__),
                response=response,
                api_type=ApiType.OPENAI,
            )

        response = await self.oai_client.completions.create(
            prompt=prepare_prompt(prompt), **args
        )
        check_for_errors(response)
        if args["stream"]:
            return await _a_process_streamed_response(
                response, options["callbacks"], chat_model_used=False
            )
        return LLMResponse(
            response.choices[0].text,
            attrs_with_normalized_usage(response.__dict__),
            response=response,
            api_type=ApiType.OPENAI,
        )

    async def load_models(self) -> dict:
        models_iter = self.oai_client.models.list()
        return {model.id: model async for model in models_iter}


class OpenAIClient(BaseAIChatClient):
    aio: "AsyncOpenAIClient"
    oai_client: openai.OpenAI | openai.AzureOpenAI
    remove_hidden_output: callable

    def __init__(self, config: Config):
        super().__init__(config)
        is_azure = config.LLM_API_PLATFORM == ApiPlatform.AZURE
        responses_mode = should_use_responses_api(config)
        # Whether the constructed client can serve the Responses API.
        # A plain openai.OpenAI client (standard OpenAI, or Azure v1 endpoint) serves
        # both Chat Completions and Responses; the classic AzureOpenAI client does not.
        self.responses_api_available = responses_mode or not is_azure
        if is_azure and responses_mode:
            client_type = openai.OpenAI
            async_client_type = openai.AsyncOpenAI
            entra_token_provider = (
                _build_azure_entra_token_provider(config)
                if config.LLM_AZURE_USE_ENTRA_ID
                else None
            )
            client_params = build_responses_client_params(
                config,
                entra_token_provider=entra_token_provider,
            )
        elif is_azure:
            client_type = openai.AzureOpenAI
            async_client_type = openai.AsyncAzureOpenAI
            if config.LLM_AZURE_USE_ENTRA_ID:
                client_params = {
                    "azure_endpoint": config.LLM_API_BASE,
                    "api_version": config.LLM_API_VERSION,
                    "azure_ad_token_provider": _build_azure_entra_token_provider(config),
                    **config.INIT_PARAMS,
                }
            else:
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
        if config.HTTP_HEADERS:
            if "default_headers" not in client_params:  # maybe set in INIT_PARAMS
                client_params["default_headers"] = {}
            client_params["default_headers"].update(config.HTTP_HEADERS)

        self.oai_client = client_type(**client_params)
        self.aio = AsyncOpenAIClient(
            oai_connection=async_client_type(
                **_async_openai_client_params(client_params)
            ),
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

    def _convert_message_content(self, message_content: TMsgContent) -> Any:
        """
        Convert the message content into a format suitable for the LLM inference chat API.
        """
        if isinstance(message_content, str):
            # Prevent conversion of string content into dict(type=text, text=...)
            # because Azure OpenAI fails with Error 400
            # when passing "azure_search" data source like following:
            # llm(..., extra_body={"data_sources"=[{"type": "azure_search",...}]})
            return message_content
        return super()._convert_message_content(message_content)

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
        if should_use_responses_api(self.config, options["use_responses_api"]):
            _ensure_responses_available(self.responses_api_available)
            return _generate_via_responses(
                self,
                prompt,
                args,
                options,
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
        return LLMResponse(
            response_text,
            attrs_with_normalized_usage(response.__dict__),
            response=response,
            api_type=ApiType.OPENAI,
        )


def _build_azure_entra_token_provider(config: Config) -> Callable[[], str]:
    """
    Build an ``azure_ad_token_provider`` callable from ``LLMConfig`` fields.

    When ``LLM_AZURE_ENTRA_CREDENTIAL`` is ``"client_secret"``, credentials are
    derived solely from ``LLM_AZURE_*`` config values. When set to ``"default"``,
    :class:`~azure.identity.DefaultAzureCredential` is used, which consults OS
    environment variables, Azure CLI cache, and other Azure SDK auto-discovery
    mechanisms.
    """
    try:
        from azure.identity import (
            DefaultAzureCredential,
            ClientSecretCredential,
            get_bearer_token_provider,
        )
    except ModuleNotFoundError as e:
        raise LLMConfigError(
            "Azure Entra ID requires the azure-identity package. "
            "Install with: pip install 'ai-microcore[azure]'"
        ) from e

    mode = (config.LLM_AZURE_ENTRA_CREDENTIAL or "default").strip().lower()
    common: dict[str, Any] = {}

    if mode == "default":
        cred = DefaultAzureCredential()
    elif mode == "client_secret":
        cred = ClientSecretCredential(
            tenant_id=config.LLM_AZURE_TENANT_ID,
            client_id=config.LLM_AZURE_CLIENT_ID,
            client_secret=config.LLM_AZURE_CLIENT_SECRET,
            **common,
        )
    else:
        raise LLMConfigError(
            f"Unknown LLM_AZURE_ENTRA_CREDENTIAL: {mode!r}. "
            "Supported modes: 'default', 'client_secret'"
        )
    return get_bearer_token_provider(cred, config.LLM_AZURE_ENTRA_SCOPE)


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
    usage = None
    last_chunk = None
    async for chunk in response:
        last_chunk = chunk
        if (chunk_usage := getattr(chunk, "usage", None)) is not None:
            usage = chunk_usage
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
    attrs = streaming_usage_attrs(usage)
    return LLMResponse(
        response_text,
        attrs=attrs,
        response=last_chunk,
        api_type=ApiType.OPENAI
    )


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
    usage = None
    last_chunk = None
    for chunk in response:
        last_chunk = chunk
        if (chunk_usage := getattr(chunk, "usage", None)) is not None:
            usage = chunk_usage
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
    attrs = streaming_usage_attrs(usage)
    return LLMResponse(
        response_text,
        attrs=attrs,
        response=last_chunk,
        api_type=ApiType.OPENAI
    )


async def _generate_via_responses_async(
    client: AsyncOpenAIClient,
    prompt: TPrompt,
    args: dict[str, Any],
    options: dict[str, Any],
    config: Config,
):
    responses_args = build_responses_request(
        prompt,
        client.sync_client.convert_prompt_to_chat_input,
        args,
    )
    response = await client.oai_client.responses.create(**responses_args)
    check_for_errors(response)
    if args.get("stream"):
        return await _a_process_streamed_response(
            adapt_responses_events_async(response),
            options["callbacks"],
            chat_model_used=True,
            hidden_output_begin=config.HIDDEN_OUTPUT_BEGIN,
            hidden_output_end=config.HIDDEN_OUTPUT_END,
        )
    response_text = extract_responses_text(response)
    if config.hiding_output():
        response_text = client.sync_client.remove_hidden_output(response_text)
    for cb in options["callbacks"]:
        if asyncio.iscoroutinefunction(cb):
            await cb(response_text)
        else:
            cb(response_text)
    return LLMResponse(
        response_text,
        attrs_with_normalized_usage({"usage": getattr(response, "usage", None)}),
        response=response,
        api_type=ApiType.OPENAI,
    )


def _generate_via_responses(
    client: "OpenAIClient",
    prompt: TPrompt,
    args: dict[str, Any],
    options: dict[str, Any],
):
    config = client.config
    responses_args = build_responses_request(
        prompt,
        client.convert_prompt_to_chat_input,
        args,
    )
    response = client.oai_client.responses.create(**responses_args)
    check_for_errors(response)
    if args.get("stream"):
        return _process_streamed_response(
            adapt_responses_events(response),
            options["callbacks"],
            chat_model_used=True,
            hidden_output_begin=config.HIDDEN_OUTPUT_BEGIN,
            hidden_output_end=config.HIDDEN_OUTPUT_END,
        )
    response_text = extract_responses_text(response)
    if config.hiding_output():
        response_text = client.remove_hidden_output(response_text)
    for cb in options["callbacks"]:
        cb(response_text)
    return LLMResponse(
        response_text,
        attrs_with_normalized_usage({"usage": getattr(response, "usage", None)}),
        response=response,
        api_type=ApiType.OPENAI,
    )


def _prepare_llm_arguments(config: Config, kwargs: dict):
    args = {**config.LLM_DEFAULT_ARGS, **kwargs}
    use_responses_api = args.pop("use_responses_api", None)
    args["model"] = args.get(
        "model",
        (
            args.get("deployment_id", config.LLM_DEPLOYMENT_ID or config.MODEL)
            if config.LLM_API_PLATFORM == ApiPlatform.AZURE
            else config.MODEL
        ),
    )
    callbacks = prepare_callbacks(config, args)
    if args.get("stream"):
        ensure_stream_include_usage(args)
    return args, {"callbacks": callbacks, "use_responses_api": use_responses_api}


def _ensure_responses_available(available: bool) -> None:
    if not available:
        raise LLMConfigError(
            "Responses API is not available for the current client. On Azure, set "
            "LLM_USE_RESPONSES_API=True so an OpenAI v1 client is built for the endpoint."
        )


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
    result = make_image_generation_response(
        images,
        save,
        {
            **response.__dict__,
            "api_type": ApiType.OPENAI,
            "response": response,
        }
    )
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
