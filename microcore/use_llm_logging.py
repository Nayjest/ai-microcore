import microcore
import microcore.ui
from colorama import Fore, Style

_orig_llm = microcore.llm


class logging_options:
    prompt_color = Fore.LIGHTGREEN_EX
    response_color = Fore.CYAN
    indent = '\t'


def logged_llm(prompt: str, **kwargs):
    nl = '\n' + logging_options.indent
    prompt_indented = nl + nl.join(prompt.split('\n'))
    print(
        f'Requesting LLM {Fore.MAGENTA}{microcore.llm_default_args["model"]}{Style.RESET_ALL}:'
        f'{logging_options.prompt_color}{prompt_indented}'
    )
    out = _orig_llm(prompt, **kwargs)
    out_indented = nl + nl.join(out.split('\n'))
    print(f'LLM Response:{logging_options.response_color}{out_indented}')
    return out


microcore.llm = logged_llm
