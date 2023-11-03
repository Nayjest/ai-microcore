from colorama import Fore
from microcore import llm


def callback(text):
    print(text, end="")


while user_msg := input(f"\n{Fore.MAGENTA}Enter message: {Fore.RESET}"):
    print(f"{Fore.MAGENTA}AI: {Fore.RESET}", end="")
    llm(user_msg, callback=callback)
