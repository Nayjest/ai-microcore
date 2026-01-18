import dataclasses
import json

from colorama import Fore, init

from .llm_backends import ApiPlatform
from .utils import is_chat_model, is_notebook
from ._env import env, config
from ._prepare_llm_args import prompt_to_message_dicts, prepare_prompt


def _serialize_message_content_blocks(content: list | str) -> str:
    """
    Serializes message content blocks into a single string
    specifically for logging.
    """
    if isinstance(content, list):
        content_str = ""
        for i, item in enumerate(content):
            num = i + 1
            if isinstance(item, str):
                item_str = item
            elif isinstance(item, dict):
                try:
                    item_str = json.dumps(
                        item,
                        ensure_ascii=False,
                        indent=2
                    )
                except TypeError:
                    item_str = repr(item)
            else:
                item_str = repr(item)
            content_str += f"[Content-Part #{num}]:\n{item_str}\n"
        if content_str.endswith("\n"):
            content_str = content_str[:-1]
        return content_str
    if not isinstance(content, str):
        content = _serialize_message_content_blocks([content])
    return content


def _format_request_log_str(prompt, **kwargs) -> str:
    nl = "\n" if LoggingConfig.DENSE else "\n" + LoggingConfig.INDENT
    model = _resolve_model(**kwargs)
    out = (
        f"{LoggingConfig.COLOR_RESET}Requesting LLM "
        f"{Fore.MAGENTA}{model}{LoggingConfig.COLOR_RESET}:"
        + (" " if LoggingConfig.DENSE else "\n")
    )
    if is_chat_model(model, env().config):
        for msg in prompt_to_message_dicts(prompt):
            role, content = (
                (msg["role"], msg["content"])
                if isinstance(msg, dict)
                else dataclasses.astuple(msg)
            )
            content = _serialize_message_content_blocks(content)
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
    if LoggingConfig.STRIP_REQUEST_LINES:
        start_lines, end_lines = LoggingConfig.STRIP_REQUEST_LINES
        max_lines = start_lines + end_lines
        lines = out.split("\n")
        if len(lines) > max_lines:
            out = "\n".join(
                lines[:start_lines]
                + [
                    f"{LoggingConfig.INDENT}{Fore.YELLOW}"
                    f"...(output was truncated)..."
                    f"{LoggingConfig.PROMPT_COLOR}"
                ]
                + (lines[-end_lines:] if end_lines else [])
            )
    return out


def _resolve_model(**kwargs):
    cfg = config()
    model = kwargs.get("model") or cfg.LLM_DEFAULT_ARGS.get("model") or cfg.MODEL
    if cfg.LLM_API_PLATFORM == ApiPlatform.AZURE:
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
    STRIP_REQUEST_LINES: tuple[int, int] | None = [40, 15]


def _log_request(prompt, **kwargs):
    LoggingConfig.OUTPUT_METHOD(_format_request_log_str(prompt, **kwargs))


def _log_response(out):
    LoggingConfig.OUTPUT_METHOD(_format_response_log_str(out))


_is_new_request = False


def _stream_log_request(prompt, **kwargs):
    global _is_new_request
    _is_new_request = True
    _log_request(prompt, **kwargs)


def _stream_log_response(out):
    global _is_new_request
    if _is_new_request:
        LoggingConfig.OUTPUT_METHOD(_format_response_log_str(out))
        _is_new_request = False
    else:
        out = f"{LoggingConfig.RESPONSE_COLOR}{out}{LoggingConfig.COLOR_RESET}"
        if not LoggingConfig.DENSE and LoggingConfig.INDENT:
            out = out.replace("\n", "\n" + LoggingConfig.INDENT)
        LoggingConfig.OUTPUT_METHOD(out)


def _print_no_nln(s):
    print(s, end='', flush=True)


def use_logging(stream: bool = False):
    """Turns on logging of LLM requests and responses to console."""
    if not is_notebook():
        init(strip=False)
    if stream:
        LoggingConfig.OUTPUT_METHOD = _print_no_nln
        # Remove non-streamable handlers if any
        if _log_request in env().llm_before_handlers:
            env().llm_before_handlers.remove(_log_request)
        if _log_response in env().llm_after_handlers:
            env().llm_after_handlers.remove(_log_response)
        # Add streamable handlers if not already present
        if _stream_log_request not in env().llm_before_handlers:
            env().llm_before_handlers.append(_stream_log_request)
        if _stream_log_response not in config().CALLBACKS:
            env().config.CALLBACKS.append(_stream_log_response)
    else:
        if _log_request not in env().llm_before_handlers:
            env().llm_before_handlers.append(_log_request)
        if _log_response not in env().llm_after_handlers:
            env().llm_after_handlers.append(_log_response)
        # Cleanup prev. streamable setup if any
        if LoggingConfig.OUTPUT_METHOD is _print_no_nln:
            LoggingConfig.OUTPUT_METHOD = print
        if _stream_log_request in env().llm_before_handlers:
            env().llm_before_handlers.remove(_stream_log_request)
        if _stream_log_response in config().CALLBACKS:
            config().CALLBACKS.remove(_stream_log_response)
