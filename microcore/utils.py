"""
General-purpose utility functions.
"""
import abc
import asyncio
import builtins
import dataclasses
import inspect
import json
import os
import sys
import re
import subprocess
import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any, Union, Callable, TYPE_CHECKING

import tiktoken
from colorama import Fore

from .message_types import MsgContentPart
from .types import BadAIAnswer, BadAIJsonAnswer
from .message_types import MsgContent, UserMsg, SysMsg, AssistantMsg
from .json_parsing import parse_json

if TYPE_CHECKING:
    from .configuration import Config


def is_chat_model(model: str, config: "Config" = None) -> bool:
    """
    Detects if model is chat model or text completion model.
    """
    if config and config.CHAT_MODE is not None:
        return config.CHAT_MODE
    completion_keywords = [
        "instruct", "davinci", "babbage", "curie", "ada",  # outdated OpenAI completion models
        "-codex",  # OpenAI Coding models, gpt-5.1-codex, gpt-5.1-codex-max, gpt-5-codex, etc.
    ]
    return not any(keyword in str(model).lower() for keyword in completion_keywords)


def is_image_model(model: str) -> bool:
    """
    Detects if model is an image generation model by name.

    Warning:
        Limited implementation: Currently only detects DALL-E and GPT-image models.

    Args:
        model: Model identifier string

    Returns:
        True if the model is identified as an image generation model

    Examples:
        >>> is_image_model("dall-e-3")
        True
        >>> is_image_model("gpt-4")
        False

    Todo:
        * Add Google / xAI / Anthropic image generation models detection.
    """
    model = str(model).lower()
    return (
        # OpenAI Image models
        model.startswith("dall-e-")
        or model.startswith("gpt-image-")
        # Google image models: gemini-2.5-flash-image, gemini-3-pro-image-preview, etc.
        or ("gemini" in model and "image" in model)
    )


class ConvertableToMessage(abc.ABC):
    """
    Trait / mixin class that provides properties to convert
    string-like objects to microcore chat message types.
    """

    def _as_message_content(self) -> str | list[dict]:
        # use conversion method from subclass if available
        if isinstance(self, MsgContent):
            return self
        if isinstance(self, MsgContentPart):
            return [self]
        return str(self)

    @property
    def as_user(self) -> UserMsg:
        return UserMsg(self._as_message_content())

    @property
    def as_system(self) -> SysMsg:
        return SysMsg(self._as_message_content())

    @property
    def as_assistant(self) -> AssistantMsg:
        return AssistantMsg(self._as_message_content())

    @property
    def as_model(self) -> AssistantMsg:
        return self.as_assistant


class ExtendedString(str):
    """
    Way of extending strings with attributes and methods.
    - Allows chaining of global functions that accept string as first argument.
      Example: `s.upper().strip()`
    - Can hold arbitrary attributes.
    - Inherits from str.
    - Provides handy utility methods for tokenization and JSON parsing.
    """

    def __new__(cls, string: str, attrs: dict = None, **kwargs):
        """
        Create new ExtendedString instance with optional attributes.
        Args:
            string (str): The string value.
            attrs (dict): Optional dictionary of attributes to set.
            **kwargs: Additional attributes to set.
        Returns:
            ExtendedString: New instance of ExtendedString.
        """
        obj = str.__new__(cls, string)
        if attrs:
            for k, v in attrs.items():
                setattr(obj, k, v)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def __getattr__(self, item):
        """
        Provides chaining of global functions.
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

    def to_tokens(
        self, for_model: str = None, encoding: str | tiktoken.Encoding = None
    ) -> list[int]:
        """
        Converts string to list of tokens.
        """
        from .tokenizing import encode

        return encode(self, for_model=for_model, encoding=encoding)

    def num_tokens(
        self, for_model: str = None, encoding: str | tiktoken.Encoding = None
    ) -> int:
        """
        Calculates quantity of tokens in target string.
        """
        return len(self.to_tokens(for_model=for_model, encoding=encoding))

    def parse_json(
        self, raise_errors: bool = True, required_fields: list[str] = None
    ) -> list | dict | float | int | str:
        return parse_json(self, raise_errors, required_fields)

    def contains_valid_json(self) -> bool:
        try:
            self.parse_json(raise_errors=True)
            return True
        except BadAIJsonAnswer:
            return False


class DataclassEncoder(json.JSONEncoder):
    """@private"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dict(dataclasses.asdict(o))
        return _default(self, o)


_default = json.JSONEncoder.default
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


def file_link(file_path: str | Path, use_file_prefix: bool = True) -> str:
    """
    Returns file name in format displayed in PyCharm console as a link.
    Args:
        file_path (str | Path): Path to the file.
        use_file_prefix (bool): Whether to add "file:///" prefix. Defaults to True
    """
    normalized_path = str(Path(file_path).absolute()).replace("\\", "/")
    return (use_file_prefix and "file:///" or "") + normalized_path


def list_files(
    target_dir: str | Path = "",
    exclude: list[str | Path] = None,
    relative_to: str | Path = None,
    absolute: bool = False,
    posix: bool = False,
) -> list[Path]:
    """
    List files in target directory, excluding those that match given fnmatch patterns.

    This function traverse target directory recursively and returns a list of all files
    that do not match the specified exclusion patterns. It can return absolute paths,
    paths relative to the target directory, or paths relative to a specified directory.

    Args:
        target_dir (str | Path): The directory to search in.
        exclude (list[str | Path]): fnmatch patterns of files to exclude.
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
    def _is_file(file_path: Path) -> bool:
        if sys.version_info >= (3, 13):
            return file_path.is_file(follow_symlinks=False)
        try:
            return file_path.is_file()
        except OSError:
            return False

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
            if _is_file(path)
            and not any(
                fnmatch(str(path.relative_to(target)), str(pattern))
                for pattern in exclude
            )
        )
    ]


def is_kaggle() -> bool:
    """
    Returns True if the code is running in a Kaggle notebook.
    """
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def is_notebook() -> bool:
    """
    Returns True if the code is running in a Jupyter notebook.
    """
    return "ipykernel" in sys.modules


def is_google_colab() -> bool:
    """
    Returns True if the code is running in a Google Colab notebook.
    """
    return "google.colab" in sys.modules


def get_vram_usage(as_string=True, color=Fore.GREEN):
    """
    Returns GPU VRAM usage as a string or a list of objects.
    """

    @dataclass
    class _MemUsage:
        name: str
        used: int
        free: int
        total: int

    cmd = (
        "nvidia-smi"
        " --query-gpu=name,memory.used,memory.free,memory.total"
        " --format=csv,noheader,nounits"
    )
    try:
        out = subprocess.check_output(cmd, shell=True, text=True).strip()

        mu = [
            _MemUsage(*[i.strip() for i in line.split(",")])
            for line in out.splitlines()
        ]
        if not as_string:
            return mu
        c, r = (color, Fore.RESET) if color else ("", "")
        return "\n".join(
            [
                f"GPU: {c}{i.name}{r}, "
                f"VRAM: {c}{i.used}{r}/{c}{i.total}{r} MiB used, "
                f"{c}{i.free}{r} MiB free"
                for i in mu
            ]
        )
    except subprocess.CalledProcessError:
        msg = "No GPU found or nvidia-smi is not installed"
        return f"{Fore.RED}{msg}{Fore.RESET}" if as_string else None


def show_vram_usage():
    """Prints GPU VRAM usage."""
    print(get_vram_usage(as_string=True))


def return_default(default, *args):
    """
    Generic-purpose default value handler.
    This utility function performs the following actions based on the type of `value`:
    - If value is an Exception type or instance, raises it.
    - If value is a function, method or built-in, calls it with provided args.
    - Otherwise, returns the value as is.
    """
    if isinstance(default, type) and issubclass(default, BaseException):
        raise default()
    if isinstance(default, BaseException):
        raise default
    if inspect.isbuiltin(default):
        return default(*args)
    if inspect.isfunction(default):
        arg_count = default.__code__.co_argcount
        if inspect.ismethod(default):
            arg_count -= 1
        return default(*args) if arg_count >= len(args) else default()

    return default


def extract_number(
    text: str,
    default=None,
    position="last",
    dtype: type | str = float,
    rounding: bool = False,
) -> int | float | Any:
    """
    Extract a number from a text.
    Returns default value if no number is found or conversion fails.
    Args:
        text (str): The input text to extract the number from.
        default: The default value to return if no number is found or conversion fails.
        position (str): "last" to extract the last number, "first" for the first number.
        dtype (type | str): The desired type of the extracted number ("int" or "float").
        rounding (bool): If True and dtype is float, rounds the result to the nearest integer.
    """
    assert position in ["last", "first"], f"Invalid position: {position}"
    idx = {"last": -1, "first": 0}[position]

    dtype = {"int": int, "float": float}.get(dtype, dtype)
    assert dtype in [int, float], f"Invalid dtype: {dtype}"
    if rounding:
        dtype = float
    regex = {int: r"[-+]?\d+", float: r"[-+]?\d*\.?\d+"}[dtype]

    numbers = re.findall(regex, str(text))
    if numbers:
        try:
            value = dtype(numbers[idx].strip())
            return round(value) if rounding else value
        except (ValueError, OverflowError):
            ...
    return return_default(default, text)


def dedent(text: str) -> str:
    """
    Remove minimal shared leading whitespace from each line
    and strip leading and trailing empty lines.
    @deprecated: Use textwrap.dedent from standard library instead.
    """
    lines = text.splitlines()
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        min_indent = min((len(line) - len(line.lstrip())) for line in non_empty_lines)
        dedented_lines = [
            line[min_indent:] if line and len(line) >= min_indent else line
            for line in lines
        ]
    else:
        dedented_lines = lines
    return "\n".join(dedented_lines)


RETURN_EXCEPTION = object()


async def run_parallel(
    tasks: list,
    max_concurrent_tasks: int,
    allow_failures: bool = False,
    return_on_failure: Any = RETURN_EXCEPTION,
    log_errors: bool = True,
):
    """
    Run tasks in parallel with a limit on the number of concurrent tasks.
    Args:
        tasks (list): A list of awaitable tasks to run.
        max_concurrent_tasks (int): Maximum number of tasks to run concurrently.
        allow_failures (bool): If True, exceptions in tasks are caught and
            handled according to the `return_on_failure` parameter.
        return_on_failure (Any): Value to return for failed tasks if `allow_failures` is True.
            If set to microcore.utils.RETURN_EXCEPTION, the exception object is returned.
        log_errors (bool): If True, errors are logged when `allow_failures` is True.
    """
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def worker(task):
        async with semaphore:
            try:
                return await task
            except Exception as e:
                if allow_failures:
                    if log_errors:
                        logging.error(e)
                    return e if return_on_failure == RETURN_EXCEPTION else return_on_failure
                raise

    return await asyncio.gather(*[worker(task) for task in tasks])


class CantResolveCallable(ValueError):
    """
    Raised when a callable cannot be resolved by name.
    """
    def __init__(self, message: str = None, name: str = None, e: Exception = None):
        message = message or f"Can't resolve callable by name '{name}'{', ' + str(e) if e else ''}"
        super().__init__(message)
        self.name = name


@lru_cache
def _load_callable(fn_path: str) -> callable:
    try:
        if "." not in fn_path:
            fn = globals()[fn_path]
        else:
            parts = fn_path.split(".")
            # Try resolve as *module.ClassName.static_method if 1st character is upper-cased
            if len(parts) >= 3 and len(parts[-2]) and parts[-2][0].isupper():
                module_name = ".".join(parts[:-2])
                class_name = parts[-2]
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    cls = getattr(module, class_name)
                    fn = getattr(cls, parts[-1])
                    assert callable(fn)
                    return fn
                except (ImportError, AttributeError, AssertionError, ValueError):
                    pass
            module_name = ".".join(parts[:-1])
            func_name = parts[-1]
            if not module_name:
                raise CantResolveCallable(f"Invalid module name: {module_name}")
            module = __import__(module_name, fromlist=[func_name])
            fn = getattr(module, func_name)
        assert callable(fn)
    except (ImportError, AttributeError, AssertionError, ValueError) as e:
        raise CantResolveCallable(name=fn_path, e=e) from e
    return fn


def resolve_callable(
    fn: Union[Callable, str, None], allow_empty=False
) -> Union[Callable, None]:
    """
    Resolves a callable function from a string.
    Supported formats:
      - module[.submodules].function
      - module[.submodules].ClassName.static_method
    """
    if callable(fn):
        return fn
    if not fn:
        if allow_empty:
            return None
        raise CantResolveCallable("Can't resolve callable: function is not specified")
    return _load_callable(fn)


def levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    The **Levenshtein distance** is the minimum number of single‑character
    edits (insertions, deletions, or substitutions) required to transform one
    string into the other.

    This implementation uses the classic Wagner–Fischer dynamic‑programming
    algorithm and stores only a single row of the DP matrix at any time,
    reducing memory usage to be linear in the length of the shorter string.

    Args:
        a (str): First input string.
        b (str): Second input string.

    Returns:
        int: Non‑negative integer representing the edit distance. A value of
        ``0`` means the strings are identical.

    Complexity:
        * **Time** ``O(ab)``
        * **Space** ``O(min(a, b))``

    Examples:
        >>> levenshtein("kitten", "sitting")
        3
        >>> levenshtein("graph", "giraffe")
        4
    """
    if a == b:
        return 0
    # Ensure a is the shorter string to reduce memory
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for i, ch_b in enumerate(b, start=1):
        current = [i]
        for j, ch_a in enumerate(a, start=1):
            cost = 0 if ch_a == ch_b else 1
            current.append(
                min(
                    current[-1] + 1,        # insertion
                    previous[j] + 1,        # deletion
                    previous[j - 1] + cost  # substitution
                )
            )
        previous = current
    return previous[-1]


def most_similar(
    needle: str,
    haystack: list[str],
    distance_func: callable = levenshtein,
    case_sensitive: bool = False,
) -> tuple[str, int]:
    """
    Find the most similar string from a list of strings using the
    specified distance function.

    Args:
        needle (str): The word to compare against.
        haystack (list[str]): A list of words to compare with.
        distance_func (callable): The distance function to use for comparison.
            Defaults to levenshtein.
        case_sensitive (bool): If True, the comparison is case-sensitive.

    Returns:
        tuple[str, int]: A tuple containing the most similar word and its distance
            from the given word.

    Raises:
        ValueError: If haystack is empty.
    """
    if not haystack:
        raise ValueError("Haystack cannot be empty")

    min_dist = float('inf')
    most_similar_word = None
    a = needle if case_sensitive else needle.lower()
    for word in haystack:
        b = word if case_sensitive else word.lower()
        dist = distance_func(a, b)
        if dist < min_dist:
            min_dist = dist
            most_similar_word = word

    return most_similar_word, min_dist


# XML tag extraction

_TAG_PATTERN = re.compile(r'<(\w+)(.*?)>(.*?)</\1>', re.DOTALL)
_TAG_ATTRIBUTE_PATTERN = re.compile(
    r'(\w+)\s?=\s?["\']([^"\']*)["\']|(\w+)\s?=\s?([^\s\'\"\>]+)'
)
TExtractedTag = tuple[str, dict[str, str], str]


def extract_tags(
    text: str,
    strip: bool = False
) -> list[TExtractedTag]:
    """
    Extract all XML tags with attributes and content.
    Returns:
         list of tuples: (tag_name, attributes_dict, content)
    """
    tags = re.findall(_TAG_PATTERN, text)
    return [
        (name, _parse_tag_attributes(attrs_str), content.strip() if strip else content)
        for name, attrs_str, content in tags
    ]


def _parse_tag_attributes(attrs_str: str) -> dict[str, str]:
    attr_dict = {}
    for match in re.finditer(_TAG_ATTRIBUTE_PATTERN, attrs_str):
        if match.group(1):  # Quoted attribute
            attr_dict[match.group(1)] = match.group(2)
        else:  # Unquoted attribute
            attr_dict[match.group(3)] = match.group(4)
    return attr_dict
