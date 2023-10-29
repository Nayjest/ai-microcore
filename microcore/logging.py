import dataclasses
import microcore
import microcore.ui
from colorama import Fore, Style
import microcore.prepare_llm_args
from .utils import is_chat_model


class LoggingConfig:
    PROMPT_COLOR = Fore.LIGHTGREEN_EX
    RESPONSE_COLOR = Fore.CYAN
    INDENT: str = '\t'
    DENSE: bool = False


def _log_request(prompt, **kwargs):
    nl = '\n' if LoggingConfig.DENSE else '\n' + LoggingConfig.INDENT
    model = _resolve_model(**kwargs)
    print(
        f'Requesting LLM {Fore.MAGENTA}{model}{Style.RESET_ALL}:',
        end=' ' if LoggingConfig.DENSE else '\n'
    )
    if is_chat_model(model):
        for msg in microcore.prepare_llm_args.prepare_chat_messages(prompt):
            role, content = (
                msg['role'],
                msg['content']
            ) if isinstance(msg, dict) else dataclasses.astuple(msg)
            nl2 = '\n' if LoggingConfig.DENSE else nl + LoggingConfig.INDENT
            content = (' ' if LoggingConfig.DENSE else nl2) + nl2.join(content.split('\n'))
            print(
                f'{"" if LoggingConfig.DENSE else LoggingConfig.INDENT}'
                f'{LoggingConfig.PROMPT_COLOR}[{role.capitalize()}]:{content}'
            )
    else:
        lines = microcore.prepare_llm_args.prepare_prompt(prompt).split('\n')
        print(
            LoggingConfig.PROMPT_COLOR
            + (' ' if LoggingConfig.DENSE else LoggingConfig.INDENT) + nl.join(lines)
        )


def _resolve_model(**kwargs):
    return (kwargs.get('model')
            or microcore.env().config.LLM_DEFAULT_ARGS.get('model')
            or microcore.env().config.MODEL)


def _log_response(out):
    nl = '\n' if LoggingConfig.DENSE else '\n' + LoggingConfig.INDENT
    out_indented = (' ' if LoggingConfig.DENSE else nl) + nl.join(out.split('\n'))
    print(f'LLM Response:{LoggingConfig.RESPONSE_COLOR}{out_indented}')


def use_logging():
    if _log_request not in microcore.env().llm_before_handlers:
        microcore.env().llm_before_handlers.append(_log_request)
    if _log_response not in microcore.env().llm_after_handlers:
        microcore.env().llm_after_handlers.append(_log_response)
