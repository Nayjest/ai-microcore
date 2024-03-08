"""
ai_module: ai_func
descr: Allows to describe python functions for LLM
"""

import ast
import inspect
from typing import Dict, Any
import docstring_parser
from .. import tpl


def func_arg_comments(func):
    func_source = inspect.getsource(func)
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


def describe_ai_func(func):
    metadata = func_metadata(func)
    return tpl("python_ai_func.j2", **metadata)
