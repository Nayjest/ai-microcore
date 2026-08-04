"""
MCPConnection.exec() accepts a tool call in any form an LLM may produce it:
a dict, a JSON string, or a raw LLMResponse (JSON, possibly in a markdown block).
The tool name is taken from the "call" field (AI_SYNTAX_FUNCTION_NAME_FIELD);
the remaining fields become the tool arguments.

This is what allows connecting MCP tools to LLM backends without native
tool-calling support: the model's text output is passed to exec() as is.
"""

import pytest

import microcore as mc
from microcore.mcp import MCPConnection, WrongMcpUsage
from microcore.wrappers.llm_response_wrapper import LLMResponse


class StubClient:
    """Records the tool call instead of hitting a real MCP server."""

    def __init__(self):
        self.called_with = None

    async def call_tool(self, name, arguments, **kwargs):
        self.called_with = (name, arguments)


@pytest.fixture
def conn():
    mc.configure(LLM_API_TYPE=mc.ApiType.NONE, USE_DOT_ENV=False)
    connection = MCPConnection()
    connection._client = StubClient()
    return connection


async def test_exec_accepts_dict(conn):
    await conn.exec({"call": "search", "query": "python"})
    assert conn._client.called_with == ("search", {"query": "python"})


async def test_exec_accepts_json_string(conn):
    await conn.exec('{"call": "search", "query": "python"}')
    assert conn._client.called_with == ("search", {"query": "python"})


async def test_exec_accepts_raw_llm_response(conn):
    llm_output = LLMResponse('```json\n{"call": "search", "query": "python"}\n```')
    await conn.exec(llm_output)
    assert conn._client.called_with == ("search", {"query": "python"})


async def test_exec_rejects_non_json_llm_response(conn):
    with pytest.raises(WrongMcpUsage):
        await conn.exec(LLMResponse("Sorry, I can't use tools."))


async def test_exec_rejects_call_without_tool_name(conn):
    with pytest.raises(WrongMcpUsage):
        await conn.exec({"query": "python"})
