"""
CLI User Interface Utilities.

This module provides a suite of helper functions for command-line interactions,
handling colored output and robust user input prompting.
"""
from colorama import Fore, Style, init
from .utils import is_notebook

if not is_notebook():
    init(strip=False)


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


def ask_yn(msg: str, default: bool | None = None) -> bool:
    """
    Prompts the user for a Yes/No confirmation via input().
    Retries indefinitely on invalid input if default value is not provided.
    Args:
        msg (str): The question to display to the user.
        default (bool | None):
            Value converted to bool and returned on invalid input or KeyboardInterrupt (Ctrl+C).
    Returns:
        bool: True if the user confirmed, False otherwise.
    """
    if default:
        yn = bright("y") + magenta + "/n"
    else:
        yn = "y/" + bright('n')
    msg += f"\nType {Fore.MAGENTA}[{yn}{Fore.MAGENTA}]{Fore.RESET}: "
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


def ask_choose(msg: str, variants: list):
    """
    Prompt the user to choose one of the variants via input() and return the chosen item.

    Args:
        msg: The prompt message to display.
        variants: A list of options to choose from.

    Returns:
        The selected element from the `variants` list.
    """
    idx = 0
    if isinstance(variants, list):
        for item in variants:
            idx += 1
            print(f"\t{Fore.MAGENTA}{idx}:{Fore.RESET}\t{item}")
    while True:
        i = input(f"{msg} {Fore.MAGENTA}[1-{len(variants)}]{Fore.RESET}: ").strip()
        if not i.isdigit():
            error("Please type a number")
            continue
        i = int(i) - 1
        if i >= len(variants) or i < 0:
            error("Incorrect choice")
            continue
        break

    item = variants[i]
    return item


def ask_non_empty(msg) -> str:
    while True:
        i = input(msg)
        if i.strip():
            break
        error("Empty input")
    return i


class _ColorFunc(str):
    """
    A hybrid string/callable class for ANSI color codes.

    Allows usage as a string for concatenation:
        `print(red + "Text")`
    Or as a wrapper function:
        `print(red("Text"))` # Automatically resets color after
    """
    def __new__(cls, code, reset_code=Fore.RESET):
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
