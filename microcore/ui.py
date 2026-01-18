"""
CLI User Interface Utilities.

This module provides a suite of helper functions for command-line interactions,
handling colored output and robust user input prompting.
"""
from typing import Union

from colorama import Fore, Style, init

from .utils import is_notebook


if not is_notebook():
    init(strip=False)


class _ColorFunc(str):
    """
    A hybrid string/callable class for ANSI color codes.

    Allows usage as a string for concatenation:
        `print(red + "Text")`
    Or as a wrapper function:
        `print(red("Text"))` # Automatically resets color after
    """
    def __new__(cls, code, reset_code=Fore.RESET) -> Union["_ColorFunc", str]:
        obj = str.__new__(cls, code)
        obj.code = code
        obj.reset_code = reset_code
        return obj

    def __call__(self, *args):
        return f"{self.code}{''.join([str(i) for i in args])}{self.reset_code}"


# Define text colors and styles
red = _ColorFunc(Fore.RED)
green = _ColorFunc(Fore.GREEN)
blue = _ColorFunc(Fore.BLUE)
cyan = _ColorFunc(Fore.CYAN)
yellow = _ColorFunc(Fore.YELLOW)
magenta = _ColorFunc(Fore.MAGENTA)
white = _ColorFunc(Fore.WHITE)
gray = _ColorFunc(Fore.LIGHTBLACK_EX)
reset = _ColorFunc(Style.RESET_ALL, "")
reset_color = _ColorFunc(Fore.RESET, "")
bright = _ColorFunc(Style.BRIGHT, Style.NORMAL)
dim = _ColorFunc(Style.DIM, Style.NORMAL)
normal = _ColorFunc(Style.NORMAL, "")

DEFAULT_QUESTION_STYLE = f"{bright}{magenta('Q: ')}"


def info(*args, color=Fore.LIGHTYELLOW_EX, **kwargs):
    """Print info message (default color: light yellow)."""
    print(*[color + str(i) for i in args], Fore.RESET, **kwargs)


def debug(*args, **kwargs):
    """Print debug message (color: blue)."""
    info(*args, color=Fore.BLUE, **kwargs)


def error(*args, **kwargs):
    """Print error message (color: red)."""
    info(*args, color=Fore.RED, **kwargs)


def warning(*args, **kwargs):
    """Print warning message (color: yellow)."""
    info(*args, color=Fore.YELLOW, **kwargs)


def ask_yn(
    msg: str,
    default: bool | None = None,
    question_style: str = DEFAULT_QUESTION_STYLE,
) -> bool:
    """
    Prompts the user for a Yes/No confirmation via input().
    Retries indefinitely on invalid input if default value is not provided.
    Args:
        msg (str): The question to display to the user.
        default (bool | None):
            Value converted to bool and returned on invalid input or KeyboardInterrupt (Ctrl+C).
        question_style (str): Style applied to the question text.
    Returns:
        bool: True if the user confirmed, False otherwise.
    """
    if default:
        yn = bright("y") + magenta + "/n"
    else:
        yn = "y/" + bright('n')
    if question_style:
        msg = f"{question_style}{msg}"
    msg += f"\n{reset}Type {magenta}[{yn}{magenta}]{reset_color}: "
    while True:
        try:
            input_val = input(msg).lower().strip()
            if any(input_val.startswith(i) for i in [
                "y", "si", "так", "да", "1", "+", "ok", "ja", "oui"
            ]):
                return True
            if any(input_val.startswith(i) for i in ["n", "0", "-", "н"]):
                return False
            if default is not None:
                warning("Incorrect input, using default:", "Yes" if default else "No")
                return bool(default)
            warning("Please type 'y' or 'n'")
        except KeyboardInterrupt:
            warning("Interrupted, using default:", "Yes" if default else "No")
            return bool(default)


def ask_choose(
    msg: str,
    variants: list | dict,
    choice_prompt: str = "Enter choice",
    question_style: str = DEFAULT_QUESTION_STYLE,
    default: str | None = None,
):
    """
    Prompt the user to choose one of the variants via input() and return the chosen item.

    Args:
        msg (str): The question displayed above the list of choices.
        variants (list[str] | dict(str|str)):
            A list or dict of options to choose from.
            If a dict is provided, dict keys will be used as return values
            and dict values will be displayed.
        choice_prompt (str): Text shown before the input cursor (default: "Enter choice").
        question_style (str): Style applied to the question text.
        default: str | None:
            Default choice returned on empty input. Must be one of the `variants`.
    Returns:
        The selected element from the `variants` list.
    Raises:
        ValueError: If the default choice is not None and not in the variants.
        ValueError: If variants is neither a list nor a dict.
    """
    def print_choice(number: int, title: str):
        text = f"  {magenta}{dim}[{reset}{magenta}{number}{dim}]{reset}  {title}"
        if number == default_idx:
            text += reset + yellow(" [default]")
        print(text)

    print(f"{question_style}{msg}{reset}" if question_style else msg)
    idx = 0
    default_idx: int | None = None
    if isinstance(variants, dict):
        keys, display_items = list(variants.keys()), list(variants.values())
        variants = keys
    elif isinstance(variants, list):
        keys = display_items = variants
    else:
        raise ValueError("Variants must be a list or a dict")

    if default is not None and default not in keys:
        raise ValueError("Default choice is not in variants list")

    for key, display in zip(keys, display_items):
        idx += 1
        if key == default:
            default_idx = idx
        print_choice(idx, display)

    choice_prompt = (choice_prompt.rstrip() + " ") if choice_prompt else ""
    str_range = f"{magenta}[{bright}1-{len(variants)}"
    if default_idx is not None:
        str_range += f"{dim}, default={default_idx}"
    str_range += f"{reset}{magenta}]{reset}"

    while True:
        i = input(f"{reset}{choice_prompt}{str_range}: ").strip()
        if not i and default is not None:
            warning("Using default choice:", str(default))
            return default
        if not i.isdigit():
            error("Please type a number")
            continue
        i = int(i) - 1
        if i >= len(variants) or i < 0:
            error("Incorrect choice")
            continue
        return variants[i]


def ask_non_empty(msg, question_style: str = DEFAULT_QUESTION_STYLE) -> str:
    """
    Prompt the user for non-empty input via input().
    Retries indefinitely until a non-empty value is provided.
    Args:
        msg (str): The question to display to the user.
        question_style (str): Style applied to the question text.
    Returns:
        str: The user's non-empty input.
    """
    while True:
        value = ask(msg, question_style)
        if value.strip():
            break
        error("Please provide a value")
    return value


def ask(msg, question_style: str = DEFAULT_QUESTION_STYLE) -> str:
    """
    Prompt the user for input via input().
    Args:
        msg (str): The question to display to the user.
        question_style (str): Style applied to the question text.
    Returns:
        str: The user's input.
    """
    msg = f"{question_style}{msg}{reset}" if question_style else msg
    if not msg.endswith(" "):
        msg += " "
    return input(msg)
