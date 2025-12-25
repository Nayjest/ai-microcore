import mimetypes
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .utils import file_link, is_google_colab, is_kaggle, is_notebook, ConvertableToMessage
from .message_types import MsgContentPart, MsgMultipartContent


def image_format_to_mime_type(output_format: str) -> str | None:
    """Convert image format to MIME type."""
    mime_types = {
        'png': 'image/png',
        'webp': 'image/webp',
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg',  # in case you also handle 'jpg'
    }
    return mime_types.get(output_format.lower().lstrip('.'), None)


class ImageInterface(MsgContentPart, ConvertableToMessage, ABC):

    @abstractmethod
    def bytes(self) -> bytes | None:
        raise NotImplementedError()

    @abstractmethod
    def mime_type(self) -> str | None:
        raise NotImplementedError()

    def extension(self) -> str:
        """Get the file extension based on the MIME type."""
        mime_type = self.mime_type()
        if mime_type:
            ext = mimetypes.guess_extension(mime_type)
            if ext:
                return ext
        return ""

    def display(self, **kwargs):
        """Display the generated image if possible."""
        if is_kaggle() or is_google_colab() or is_notebook():
            from IPython.display import display, Image as IPythonImage
            display(IPythonImage(data=self.bytes(), **kwargs))
        else:
            print(repr(self))
        return self

    def __repr__(self):
        return "<Image>"


class ImageListInterface(MsgMultipartContent, ConvertableToMessage, ABC):
    @abstractmethod
    def images(self) -> list[ImageInterface]:
        raise NotImplementedError()

    def parts(self) -> list[MsgContentPart]:
        return self.images()


class ImageList(ImageListInterface, list):
    def images(self) -> list[ImageInterface]:
        return list(self)


@dataclass
class FileImage(ImageInterface):

    file: str | None
    _bytes: bytes | None = None
    _mime: str | None = None

    def bytes(self):
        if self.file and self._bytes is None:
            with open(self.file, "rb") as f:
                self._bytes = f.read()
        return self._bytes

    def mime_type(self) -> str | None:
        if self.file and self._mime is None:
            self._mime, _ = mimetypes.guess_type(self.file)
        return self._mime

    def __repr__(self):
        return f"<FileImage {file_link(self.file)}>"


class FileImageList(ImageListInterface):
    files: list[str] | None

    def images(self) -> list[FileImage]:
        if not self.files:
            return []
        return [FileImage(file=f) for f in self.files]


class Image(ImageInterface):
    def __init__(
        self,
        data: bytes,
        mime_type: str | None = None
    ):
        self._bytes = data
        self._mime_type = mime_type

    def bytes(self):
        return self._bytes

    def mime_type(self) -> str | None:
        return self._mime_type

    def store(self, file_path: str) -> str:
        from .file_storage import storage
        actual_fn = storage.write(file_path, self.bytes(), rewrite_existing=False)
        actual_fn = storage.abs_path(actual_fn)
        logging.info(f"Image saved to {file_link(actual_fn)}")
        return actual_fn
