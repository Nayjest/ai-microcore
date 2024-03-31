import builtins
import dataclasses
import inspect
import json
import os
import sys
import re
from fnmatch import fnmatch
from pathlib import Path

from .types import BadAIAnswer


def is_chat_model(model: str) -> bool:
    """Detects if model is chat model or text completion model"""
    completion_keywords = ["instruct", "davinci", "babbage", "curie", "ada"]
    return not any(keyword in model.lower() for keyword in completion_keywords)


class ExtendedString(str):
    """
    Provides a way of extending string with attributes and methods
    """

    def __new__(cls, string: str, attrs: dict = None):
        """
        Allows string to have attributes.
        """
        obj = str.__new__(cls, string)
        if attrs:
            for k, v in attrs.items():
                setattr(obj, k, v)
        return obj

    def __getattr__(self, item):
        """
        Provides chaining of global functions
        """
        global_func = inspect.currentframe().f_back.f_globals.get(item) or vars(
            builtins
        ).get(item, None)
        if callable(global_func):

            def method_handler(*args, **kwargs):
                res = global_func(self, *args, **kwargs)
                if isinstance(res, str) and not isinstance(res, ExtendedString):
                    res = ExtendedString(res)
                return res

            return method_handler

        # If there's not a global function with that name, raise an AttributeError as usual
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )


class DataclassEncoder(json.JSONEncoder):
    """@private"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


json.JSONEncoder.default = DataclassEncoder().default


def parse(
    text: str, field_format: str = r"\[\[(.*?)\]\]", required_fields: list = None
) -> dict:
    """
    Parse a document divided into sections and convert it into a dictionary.
    """
    pattern = rf"{field_format}\n(.*?)(?=\n{field_format}|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    result = {key.strip().lower(): value for key, value, _ in matches}
    if required_fields:
        for field in required_fields:
            if field not in result:
                raise BadAIAnswer(f"Field '{field}' is required but not found")
    return result


def file_link(file_path: str | Path):
    """Returns file name in format displayed in PyCharm console as a link."""
    return "file:///" + str(Path(file_path).absolute()).replace("\\", "/")


def list_files(
    target_dir: str | Path = "",
    exclude: list[str | Path] = None,
    relative_to: str | Path = None,
    absolute: bool = False,
    posix: bool = False,
) -> list[Path]:
    """
    Lists files in a specified directory, excluding those that match given patterns.

    This function traverses the specified directory recursively and returns a list of all files
    that do not match the specified exclusion patterns. It can return absolute paths,
    paths relative to the target directory, or paths relative to a specified directory.

    Args:
        target_dir (str | Path): The directory to search in.
        exclude (list[str | Path]): Patterns of files to exclude.
        relative_to (str | Path, optional): Base directory for relative paths.
            If None, paths are relative to `target_dir`. Defaults to None.
        absolute (bool, optional): If True, returns absolute paths. Defaults to False.
        posix (bool, optional): If True, returns posix paths. Defaults to False.

    Returns:
        list[Path]: A list of Path objects representing the files found.

    Example:
        exclude_patterns = ['*.pyc', '__pycache__/*']
        target_directory = '/path/to/target'
        files = list_files(target_directory, exclude_patterns)
    """
    exclude = exclude or []
    target = Path(target_dir or os.getcwd()).resolve()
    relative_to = Path(relative_to).resolve() if relative_to else None
    if absolute and relative_to is not None:
        raise ValueError(
            "list_files(): Cannot specify both 'absolute' and 'relative_to'. Choose one."
        )
    return [
        p.as_posix() if posix else p
        for p in (
            path.resolve() if absolute else path.relative_to(relative_to or target)
            for path in target.rglob("*")
            if path.is_file()
            and not any(
                fnmatch(str(path.relative_to(target)), str(pattern))
                for pattern in exclude
            )
        )
    ]


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def is_notebook() -> bool:
    return "ipykernel" in sys.modules
