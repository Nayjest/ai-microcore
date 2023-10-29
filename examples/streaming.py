from microcore import llm
from colorama import Fore


def callback(text):
    print(text, end='')


while user_msg := input(f'\n{Fore.MAGENTA}Enter message: {Fore.RESET}'):
    print(f'{Fore.MAGENTA}AI: {Fore.RESET}', end='')
    llm(user_msg, callback=callback)
