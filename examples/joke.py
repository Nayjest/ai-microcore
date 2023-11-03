from colorama import Fore
from microcore import llm

llm(
    "Tell me a joke",
    temperature=0.7,
    callback=lambda x: print(Fore.LIGHTCYAN_EX + x, end=""),
)
print("\n")
