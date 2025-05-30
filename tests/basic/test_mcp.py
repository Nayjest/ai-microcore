import os
import microcore as mc
import pytest

servers_cfg = [
    {"name": "mcp1", "url": "http://localhost:8000"},
    {"name": "mcp2", "url": "http://localhost:9000"},
]


def test_mcp_configuring():
    mc.configure(
        LLM_API_KEY="123",
        MCP_SERVERS=servers_cfg,
    )
    mcp = mc.mcp_server("mcp1")
    assert mcp.name == "mcp1"
    assert mcp.url == "http://localhost:8000"

    mcp = mc.mcp.server("mcp2")
    assert mcp.name == "mcp2"
    assert mcp.url == "http://localhost:9000"

    with pytest.raises(ValueError):
        mc.mcp_server("mcp3")


def test_custom_mcp_registry():
    assert mc.mcp.MCPRegistry(servers_cfg).get("mcp1").name == "mcp1"


def test_mcp_from_env():
    os.environ["MCP_SERVERS"] = '[{"name": "e_mcp", "url": "ws://1.1.1.1"}]'
    mc.configure(VALIDATE_CONFIG=False)
    assert mc.mcp.server("e_mcp").name == "e_mcp"
    del os.environ["MCP_SERVERS"]


def test_create_server():
    server = mc.mcp.MCPServer(url="http://localhost:8000", name="test_server")
    assert server.name == "test_server"
    assert server.url == "http://localhost:8000"
    assert len(server.tools) == 0


def test_tool():
    mc.configure(VALIDATE_CONFIG=False)
    tool = mc.mcp.Tool(
        name="test_tool",
        description="A test tool",
        args={
            "arg": {
                "name": "arg",
                "type": "string",
                "description": "A test argument"
            }
        }

    )
    assert tool.args["arg"].name == "arg"
    assert tool.args["arg"].type == "string"
    assert tool.args["arg"].description == "A test argument"
    assert tool.args["arg"].required is True

    serialized = str(tool)
    assert mc.config().AI_SYNTAX_FUNCTION_NAME_FIELD in serialized
    assert tool.name in serialized
    assert tool.description in serialized
    assert tool.args["arg"].name in serialized
    assert tool.args["arg"].description in serialized
    assert tool.args["arg"].type in serialized


@pytest.mark.asyncio
async def test_mcp_time():
    mc.configure(MCP_SERVERS=['https://time.mcp.inevitable.fyi/mcp'], VALIDATE_CONFIG=False)
    mcp: mc.mcp.MCPConnection = await mc.mcp_server("time.mcp.inevitable.fyi").connect()
    assert "get_current_time" in mcp.tools
    res = (await mcp.exec(dict(
        call="get_current_time",
        timezone="UTC",
    ))).parse_json()
    assert "datetime" in res
    assert res["timezone"] == "UTC"
    await mcp.close()


@pytest.mark.asyncio
async def test_mcp_precache():
    mc.configure(MCP_SERVERS=['https://time.mcp.inevitable.fyi/mcp'], VALIDATE_CONFIG=False)
    await mc.mcp.precache_tools()
    assert len(mc.mcp_server("time.mcp.inevitable.fyi").get_tools_cache()) >= 1


@pytest.mark.asyncio
async def test_mcp_update_tools_cache():
    mc.configure(MCP_SERVERS=[dict(
        name='test2',
        url='https://time.mcp.inevitable.fyi/mcp'
    )], VALIDATE_CONFIG=False)
    mcp = await mc.mcp.server('test2').connect(fetch_tools=False)
    await mcp.fetch_tools()
    await mcp.update_tools_cache()
