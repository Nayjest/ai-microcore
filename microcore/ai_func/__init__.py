"""
ai_module: ai_func
descr: Allows to describe python functions for LLM
"""

import ast
import inspect
from enum import Enum
from typing import Dict, Any
import docstring_parser
from ..utils import dedent
from .._env import env

class AiFuncSyntax(str, Enum):
    PYTHONIC: str = "pythonic"
    JSON: str = "json"
    DEFAULT: str = str(JSON)

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


def func_metadata(func) -> Dict[str, Any]:
    metadata = dict(
        name=func.__name__,
        description=inspect.getdoc(func),
        args={},
    )

    for name, param in inspect.signature(func).parameters.items():
        # Store parameter info
        metadata["args"][name] = {"kind": str(param.kind)}
        if param.default != param.empty:
            if isinstance(param.default, str):
                metadata["args"][name]["default"] = '"' + param.default + '"'
            else:
                metadata["args"][name]["default"] = param.default
        else:
            metadata["args"][name]["default"] = "NOT_SET"
        if param.annotation != param.empty:
            param_type = (
                str(param.annotation)
                .replace("typing.", "")
                .replace("<class '", "")
                .replace("'>", "")
            )
            metadata["args"][name]["type"] = param_type

    arg_comments = func_arg_comments(func)
    for name, val in metadata["args"].items():
        val["comment"] = arg_comments[name]

    # Parse docstring
    parsed_docstring = docstring_parser.parse(inspect.getdoc(func))

    # Add descriptions from parsed docstring to parameters
    for param in parsed_docstring.params:
        if param.arg_name in metadata["args"]:
            metadata["args"][param.arg_name]["docstr"] = param.description

    return metadata


def describe_ai_func(func: callable, syntax: AiFuncSyntax | str = None) -> str:
    """
    Renders function description for LLM
    Args:
        func: callable: function to describe
        syntax: AiFuncSyntax | str: syntax to use for the description
                - Use AiFuncSyntax enums to use standard templates (""json", "pythonic")
                - Use custom template name to use custom template
    Returns: str: rendered description, part of prompt
    """
    syntax = syntax or AiFuncSyntax.DEFAULT
    tpl_file = f"ai-func.{syntax}.j2" if syntax in AiFuncSyntax else syntax
    metadata = func_metadata(func)
    return env().tpl_function(tpl_file, **metadata)
