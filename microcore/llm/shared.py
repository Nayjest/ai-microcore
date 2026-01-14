import re

from ..images import FileImage, Image
from ..wrappers.llm_response_wrapper import ImageGenerationResponse, StoredImageGenerationResponse
from ..configuration import Config


def make_remove_hidden_output(config: Config) -> callable:
    pattern = re.compile(
        f"{config.HIDDEN_OUTPUT_BEGIN}.*?{config.HIDDEN_OUTPUT_END}", flags=re.DOTALL
    )

    def remove_hidden_output(text: str) -> str:
        return pattern.sub("", text)

    return remove_hidden_output


def prepare_callbacks(config: Config, args, set_stream: bool = True) -> list[callable]:
    callbacks = args.pop("callbacks", []) or [] + config.CALLBACKS or []
    if "callback" in args:
        cb = args.pop("callback")
        if cb:
            callbacks.append(cb)
    if set_stream and "stream" not in args:
        args["stream"] = bool(callbacks)

    return callbacks


def make_image_generation_response(
    images: list[Image],
    save: str | bool,
    attrs: dict
) -> ImageGenerationResponse | StoredImageGenerationResponse:
    if save:
        file_name = save if isinstance(save, str) else "generated_images/image-<n>.png"
        images = [FileImage(i.store(file_name)) for i in images]
        return StoredImageGenerationResponse(images=images, **attrs)
    return ImageGenerationResponse(images=images, **attrs)
