from colorama import Fore, init

init(autoreset=True)
_input = input


def info(*args, color=Fore.LIGHTYELLOW_EX, **kwargs):
    print(*[color + i for i in args], **kwargs)


def debug(msg):
    info(msg, color=Fore.BLUE)


def error(*args, **kwargs):
    print(*[Fore.RED + i for i in args], **kwargs)


def ask_yn(msg):
    return "y" in input(msg + " (y/n)").lower().strip()


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
