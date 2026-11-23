"""Provider request shapes for portable native tools."""

from typing import Any, Callable

from ..message_types import ToolDefinition


def convert_tool_definitions(
    tools: Any,
    convert: Callable[[ToolDefinition], Any],
) -> Any:
    """Convert portable definitions while leaving provider-native entries unchanged."""
    if not isinstance(tools, list):
        return tools
    return [
        convert(tool) if isinstance(tool, ToolDefinition) else tool for tool in tools
    ]


def openai_chat_tool(tool: ToolDefinition) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def openai_responses_tool(tool: ToolDefinition) -> dict:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def anthropic_tool(tool: ToolDefinition) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def google_tool(tool: ToolDefinition) -> dict:
    return {
        "function_declarations": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        ]
    }
