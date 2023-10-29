from microcore import llm
from colorama import Fore

llm(
    'Tell me a joke',
    temperature=0.7,
    callback=lambda x: print(Fore.LIGHTCYAN_EX + x, end='')
)
print('\n')
