import dataclasses
import microcore
import microcore.ui
import microcore.openai
from colorama import Fore, Style
import microcore.prepare_llm_args

_orig_llm = microcore.llm


class cfg:
    prompt_color = Fore.LIGHTGREEN_EX
    response_color = Fore.CYAN
    indent = '\t'
    dense: bool = False


def _model(**kwargs): return kwargs.get('model') or microcore.llm_default_args.get('model') or 'gpt-3.5-turbo'


def _logged_llm(prompt, **kwargs):
    nl = '\n' if cfg.dense else '\n' + cfg.indent
    model = _model(**kwargs)
    print(f'Requesting LLM {Fore.MAGENTA}{model}{Style.RESET_ALL}:', end=' ' if cfg.dense else '\n')
    if 'gpt' in model:
        for msg in microcore.prepare_llm_args.prepare_chat_messages(prompt):
            role, content = (msg['role'], msg['content']) if isinstance(msg, dict) else dataclasses.astuple(msg)
            nl2 = '\n' if cfg.dense else nl + cfg.indent
            content = (' ' if cfg.dense else nl2)+nl2.join(content.split('\n'))
            print(f'{"" if cfg.dense else cfg.indent}{cfg.prompt_color}[{role.capitalize()}]:{content}')
    else:
        lines = microcore.prepare_llm_args.prepare_prompt(prompt).split('\n')
        print(cfg.prompt_color + (' ' if cfg.dense else cfg.indent) + nl.join(lines))
    out = _orig_llm(prompt, **kwargs)
    out_indented = (' ' if cfg.dense else nl) + nl.join(out.split('\n'))
    print(f'LLM Response:{cfg.response_color}{out_indented}')
    return out


microcore.llm = _logged_llm
