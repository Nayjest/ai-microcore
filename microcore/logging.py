import dataclasses
from colorama import Fore, init

from .configuration import ApiType
from ._env import env, config
from ._prepare_llm_args import prepare_chat_messages, prepare_prompt
from .utils import is_chat_model, is_notebook


def _format_request_log_str(prompt, **kwargs) -> str:
    nl = "\n" if LoggingConfig.DENSE else "\n" + LoggingConfig.INDENT
    model = _resolve_model(**kwargs)
    out = (
        f"{LoggingConfig.COLOR_RESET}Requesting LLM "
        f"{Fore.MAGENTA}{model}{LoggingConfig.COLOR_RESET}:"
        + (" " if LoggingConfig.DENSE else "\n")
    )
    if is_chat_model(model, env().config):
        for msg in prepare_chat_messages(prompt):
            role, content = (
                (msg["role"], msg["content"])
                if isinstance(msg, dict)
                else dataclasses.astuple(msg)
            )
            nl2 = "\n" if LoggingConfig.DENSE else nl + LoggingConfig.INDENT
            content = (" " if LoggingConfig.DENSE else nl2) + nl2.join(
                content.split("\n")
            )
            out += (
                f"{'' if LoggingConfig.DENSE else LoggingConfig.INDENT}"
                f"[{role.capitalize()}]:"
                f"{LoggingConfig.PROMPT_COLOR}{content}{LoggingConfig.COLOR_RESET}\n"
            )
    else:
        lines = prepare_prompt(prompt).split("\n")
        out = (
            LoggingConfig.PROMPT_COLOR
            + (" " if LoggingConfig.DENSE else LoggingConfig.INDENT)
            + nl.join(lines)
            + LoggingConfig.COLOR_RESET
        )
        if out.endswith("\n"):
            out = out[:-1]
    return out


def _resolve_model(**kwargs):
    cfg = config()
    model = kwargs.get("model") or cfg.LLM_DEFAULT_ARGS.get("model") or cfg.MODEL
    if cfg.LLM_API_TYPE == ApiType.AZURE:
        model = f"azure:{model}"
    return model


def _format_response_log_str(out) -> str:
    nl = "\n" if LoggingConfig.DENSE else "\n" + LoggingConfig.INDENT
    out_indented = (" " if LoggingConfig.DENSE else nl) + nl.join(
        (out or "").split("\n")
    )
    return (
        f"{LoggingConfig.COLOR_RESET}LLM Response:"
        f"{LoggingConfig.RESPONSE_COLOR}{out_indented}{LoggingConfig.COLOR_RESET}"
    )


class LoggingConfig:
    PROMPT_COLOR = Fore.LIGHTGREEN_EX
    RESPONSE_COLOR = Fore.CYAN
    COLOR_RESET = Fore.RESET
    INDENT: str = " " * 4
    DENSE: bool = False
    OUTPUT_METHOD: callable = print
    REQUEST_FORMATTER: callable = _format_request_log_str
    RESPONSE_FORMATTER: callable = _format_response_log_str


def _log_request(prompt, **kwargs):
    LoggingConfig.OUTPUT_METHOD(_format_request_log_str(prompt, **kwargs))


def _log_response(out):
    LoggingConfig.OUTPUT_METHOD(_format_response_log_str(out))


def use_logging():
    """Turns on logging of LLM requests and responses to console."""
    if not is_notebook():
        init(autoreset=True)
    if _log_request not in env().llm_before_handlers:
        env().llm_before_handlers.append(_log_request)
    if _log_response not in env().llm_after_handlers:
        env().llm_after_handlers.append(_log_response)
