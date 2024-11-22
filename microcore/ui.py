from colorama import Fore, init
from .utils import is_notebook

if not is_notebook():
    init(autoreset=True)


def info(*args, color=Fore.LIGHTYELLOW_EX, **kwargs):
    print(*[color + i for i in args], **kwargs)


def debug(msg):
    info(msg, color=Fore.BLUE)


def error(*args, **kwargs):
    print(*[Fore.RED + str(i) for i in args], **kwargs)


def warning(*args, **kwargs):
    print(*[Fore.YELLOW + str(i) for i in args], **kwargs)


def ask_yn(msg, default=False):
    try:
        input_val = input(msg + " (y/n) ").lower().strip()
        return (
            any(i in input_val for i in ["y", "si", "так", "да", "1", "+"])
            if default
            else not any(i in input_val for i in ["n", "0", "-", "н"])
        )
    except KeyboardInterrupt:
        warning("Interrupted, using default:", "Yes" if default else "No")
        return default


def ask_choose(msg, variants: list):
    i = 0
    if isinstance(variants, list):
        for item in variants:
            i += 1
            print(f"\t{Fore.MAGENTA}{i}:{Fore.RESET}\t{item}")
    while True:
        i = input(f"{msg} {Fore.MAGENTA}[1-{len(variants)}]{Fore.RESET}: ")
        if not i.isdigit():
            error("Not a number")
            continue
        i = int(i) - 1
        if i >= len(variants) or i < 0:
            error("Incorrect number")
            continue
        break

    item = variants[int(i)]
    return item


def ask_non_empty(msg):
    while True:
        i = input(msg)
        if i.strip():
            break
        error("Empty input")
    return i


def magenta(msg):
    return f"{Fore.MAGENTA}{msg}{Fore.RESET}"


def yellow(msg):
    return f"{Fore.YELLOW}{msg}{Fore.RESET}"


def red(msg):
    return f"{Fore.RED}{msg}{Fore.RESET}"


def blue(msg):
    return f"{Fore.BLUE}{msg}{Fore.RESET}"


def green(msg):
    return f"{Fore.GREEN}{msg}{Fore.RESET}"


def cyan(msg):
    return f"{Fore.CYAN}{msg}{Fore.RESET}"


def white(msg):
    return f"{Fore.WHITE}{msg}{Fore.RESET}"


def gray(msg):
    return f"\033[90m{msg}{Fore.RESET}"


def black(msg):
    return f"{Fore.BLACK}{msg}{Fore.RESET}"
