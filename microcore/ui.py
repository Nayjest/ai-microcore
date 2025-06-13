from colorama import Fore, init
from .utils import is_notebook

if not is_notebook():
    init(autoreset=True, strip=False)


def info(*args, color=Fore.LIGHTYELLOW_EX, **kwargs):
    print(*[color + str(i) for i in args], **kwargs)


def debug(msg):
    info(msg, color=Fore.BLUE)


def error(*args, **kwargs):
    print(*[Fore.RED + str(i) for i in args], **kwargs)


def warning(*args, **kwargs):
    print(*[Fore.YELLOW + str(i) for i in args], **kwargs)


def ask_yn(msg: str, default: bool | None = None) -> bool:
    try:
        input_val = input(msg + " (y/n) ").lower().strip()
        if any(i in input_val for i in ["y", "si", "так", "да", "1", "+"]):
            return True
        if any(i in input_val for i in ["n", "0", "-", "н"]):
            return False
        if default is not None:
            return default
        warning("Please answer with y/n")
        return ask_yn(msg, default)
    except KeyboardInterrupt:
        warning("Interrupted, using default:", "Yes" if default else "No")
        return default


def ask_choose(msg: str, variants: list):
    idx = 0
    if isinstance(variants, list):
        for item in variants:
            idx += 1
            print(f"\t{Fore.MAGENTA}{idx}:{Fore.RESET}\t{item}")
    while True:
        i = input(f"{msg} {Fore.MAGENTA}[1-{len(variants)}]{Fore.RESET}: ").strip()
        if not i.isdigit():
            error("Not a number")
            continue
        i = int(i) - 1
        if i >= len(variants) or i < 0:
            error("Incorrect number")
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
    Cli output coloring function
    """
    def __new__(cls, code):
        obj = str.__new__(cls, code)
        obj.code = code
        return obj

    def __call__(self, *args):
        return f"{self.code}{''.join([str(i) for i in args])}{Fore.RESET}"


# Define colors
red = _ColorFunc(Fore.RED)
green = _ColorFunc(Fore.GREEN)
blue = _ColorFunc(Fore.BLUE)
cyan = _ColorFunc(Fore.CYAN)
yellow = _ColorFunc(Fore.YELLOW)
magenta = _ColorFunc(Fore.MAGENTA)
white = _ColorFunc(Fore.WHITE)
gray = _ColorFunc(Fore.LIGHTBLACK_EX)
reset = _ColorFunc(Fore.RESET)
