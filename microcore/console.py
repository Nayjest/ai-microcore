import os
from colorama import Fore, init
init(autoreset=True)
def print_sys(msg): print(Fore.LIGHTYELLOW_EX + msg)
def debug_log(msg): print(Fore.BLUE + msg)
def user_input(): return input().strip()
