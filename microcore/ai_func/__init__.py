"""
ai_module: ai_func
descr: Allows to describe python functions for LLM
"""

import ast
import inspect
import functools
import logging
from enum import Enum
from typing import Dict, Any, Optional, Iterable
import docstring_parser
from ..utils import dedent, extract_tags
from .._env import env
from ..wrappers.llm_response_wrapper import LLMResponse
from ..ui import yellow, blue


class AiFuncSyntax(str, Enum):
    PYTHONIC: str = "pythonic"
    JSON: str = "json"
    TAG: str = "tag"

    def __str__(self):
        return self.value


def func_arg_comments(func):
    func_source = dedent(inspect.getsource(func))
    module = ast.parse(func_source)
    func_def = module.body[0]

    # Get comments in the function's arguments
    arg_comments = {}
    for arg in func_def.args.args:
        # Get the line of code where this argument is defined
        arg_line = func_source.split("\n")[arg.lineno - 1]
        # If there's a comment in this line, get it
        if "#" in arg_line:
            comment = arg_line.split("#")[1].strip()
            arg_comments[arg.arg] = comment
        else:
            arg_comments[arg.arg] = ""

    return arg_comments


def func_metadata(func, name=None) -> Dict[str, Any]:
    metadata = dict(
        name=name or func.__name__,
        description=inspect.getdoc(func),
        args={},
    )

    for key, param in inspect.signature(func).parameters.items():
        # Store parameter info
        metadata["args"][key] = {"kind": str(param.kind)}
        if param.default != param.empty:
            if isinstance(param.default, str):
                metadata["args"][key]["default"] = '"' + param.default + '"'
            else:
                metadata["args"][key]["default"] = param.default
        else:
            metadata["args"][key]["default"] = "NOT_SET"
        if param.annotation != param.empty:
            param_type = (
                str(param.annotation)
                .replace("typing.", "")
                .replace("<class '", "")
                .replace("'>", "")
            )
            metadata["args"][key]["type"] = param_type

    arg_comments = func_arg_comments(func)
    for key, val in metadata.get("args", {}).items():
        val["comment"] = arg_comments[key]

    # Parse docstring
    parsed_docstring = docstring_parser.parse(inspect.getdoc(func))

    # Add descriptions from parsed docstring to parameters
    for param in parsed_docstring.params:
        if param.arg_name in metadata.get("args", {}):
            metadata["args"][param.arg_name]["docstr"] = param.description

    return metadata


def describe_ai_func(
    func: callable,
    syntax: AiFuncSyntax | str = None,
    name: str = None,
    render_settings: dict = None,
) -> str:
    """
    Renders function description for LLM
    Args:
        func: callable: function to describe
        syntax: AiFuncSyntax | str: syntax to use for the description
                - Use AiFuncSyntax enums to use standard templates (""json", "pythonic")
                - Use custom template name to use custom template
        name: str: override function name
        render_settings: dict: Custom variables for Jinja2 template
    Returns: str: rendered description, part of prompt
    """
    syntax = syntax or env().config.DEFAULT_AI_FUNCTION_SYNTAX
    tpl_file = f"ai-func.{syntax}.j2" if syntax in AiFuncSyntax else syntax
    metadata = func_metadata(func, name=name)
    return env().tpl_function(tpl_file, **metadata, render_settings=render_settings or {})


def ai_func(
    func: callable = None,
    *,
    syntax: AiFuncSyntax | str = None,
    name: str = None,
) -> callable:
    """
    Decorate function so when serialized to string, it returns
    its description for LLM prompts.
    """
    def decorator(f: callable) -> AIFunctionWrapper:
        return AIFunctionWrapper(f, syntax=syntax, name=name)
    if func is None:
        return decorator
    return decorator(func)


class AIFunctionWrapper:
    """
    Wrapper class that allows custom __str__
    for usage in prompts while preserving function behavior.
    """

    def __init__(
        self,
        func: callable,
        syntax: AiFuncSyntax | str = None,
        name: str = None,
        render_settings: dict = None,
    ):
        self.func = func
        self.syntax = syntax
        self.name = name
        self.render_settings = render_settings
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self) -> str:
        return describe_ai_func(
            self.func,
            syntax=self.syntax,
            name=self.name,
            render_settings=self.render_settings
        )

    def __repr__(self) -> str:
        return repr(self.func)

    def __getattr__(self, name):
        return getattr(self.func, name)


def extract_json_tool_params(
    llm_response: LLMResponse | str,
    raise_errors=False
) -> tuple[str, list, dict] | None:
    if not isinstance(llm_response, LLMResponse):
        llm_response = LLMResponse(llm_response)

    params = llm_response.parse_json(
        raise_errors=raise_errors,
        required_fields=[env().config.AI_SYNTAX_FUNCTION_NAME_FIELD],
    )
    if not params:
        return None
    fn_name = params.pop(env().config.AI_SYNTAX_FUNCTION_NAME_FIELD)
    return fn_name, [], params


def extract_tag_tool_params(
    llm_response: LLMResponse | str,
    raise_errors=False,
) -> tuple[str, list, dict] | None:
    tags = extract_tags(llm_response)
    if not tags:
        return None
    if len(tags) > 1:
        if raise_errors:
            raise ValueError("Response contains multiple tags when only one expected")
        logging.warning("Response contains multiple tags, but only the first one will be used.")
    tag, attrs, content = tags[0]
    return tag, [content], attrs


ai_response_tool_params_extractors = {
    AiFuncSyntax.JSON: extract_json_tool_params,
    AiFuncSyntax.TAG: extract_tag_tool_params,
}


def extract_tool_params(llm_response: LLMResponse | str) -> tuple[str, list, dict] | None:
    syntax = env().config.DEFAULT_AI_FUNCTION_SYNTAX
    if syntax not in ai_response_tool_params_extractors:
        raise ValueError(f"No tool call extractor defined for syntax '{syntax}'")
    return ai_response_tool_params_extractors[syntax](llm_response)


class ToolSet(list[AIFunctionWrapper]):

    def __init__(
        self,
        tools: Optional[Iterable[AIFunctionWrapper]] = None,
        verbose: bool = True,
        separator: str = "\n\n",
    ):
        self.verbose = verbose
        self.separator = separator
        if tools is None:
            super().__init__()
            return

        safe_items = []
        for item in tools:
            if not isinstance(item, AIFunctionWrapper):
                raise TypeError(
                    f"ToolSet can only contain AIFunctionWrapper instances, "
                    f"got {type(item).__name__}"
                )
            safe_items.append(item)

        super().__init__(safe_items)

    def __str__(self) -> str:
        return self.separator.join(str(func) for func in self)

    def extract_tool_params(
        self,
        llm_response: LLMResponse | str,
    ) -> tuple[str, list, dict] | None:
        tool_call_extractors = set()
        for func in self:
            syntax = func.syntax or env().config.DEFAULT_AI_FUNCTION_SYNTAX
            if syntax not in ai_response_tool_params_extractors:
                raise ValueError(f"No tool call extractor defined for syntax '{syntax}'")
            tool_call_extractors.add(ai_response_tool_params_extractors[syntax])
        for extractor in tool_call_extractors:
            if tool_call := extractor(llm_response, raise_errors=False):
                return tool_call
        return None

    def get_tool(self, tool_name: str) -> AIFunctionWrapper:
        for tool in self:
            if tool_name in (tool.func.__name__, tool.name):
                return tool
        raise ValueError(f"Function '{tool_name}' not found in ToolSet")

    def call(self, tool_name: str, args, kwargs) -> Any:
        if self.verbose:
            logging.info(
                "Calling %s with args=%s, kwargs=%s",
                yellow(tool_name),
                blue(args),
                blue(kwargs)
            )
        return self.get_tool(tool_name)(*args, **kwargs)

    async def async_call(self, tool_name: str, args, kwargs) -> Any:
        if self.verbose:
            logging.info(
                "Calling %s with args=%s, kwargs=%s",
                yellow(tool_name),
                blue(args),
                blue(kwargs)
            )
        return await (self.get_tool(tool_name)(*args, **kwargs))
