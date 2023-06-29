from colorama import Fore, init
init(autoreset=True)
_input = input


def info(*args, color=Fore.LIGHTYELLOW_EX, **kwargs): print(*[color + i for i in args], **kwargs)
def debug(msg): info(msg, color=Fore.BLUE)
def input(*args, **kwargs): return _input(*args, **kwargs).strip()
def ask_yn(msg): return 'y' in input(msg + ' (y/n)').lower()
