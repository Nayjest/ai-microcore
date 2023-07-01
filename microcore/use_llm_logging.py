import microcore
import microcore.ui
from colorama import Fore

_orig_llm = microcore.llm


def logged_llm(prompt: str, **kwargs):
    print(f'Requesting LLM:\n{Fore.LIGHTGREEN_EX}{prompt}')
    out = _orig_llm(prompt, **kwargs)
    print(f'LLM Answer: {Fore.CYAN}{out}')
    return out


microcore.llm = logged_llm