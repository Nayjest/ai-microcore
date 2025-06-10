import os
import microcore as mc
import pytest
from microcore.mcp import ToolsCache


@pytest.mark.asyncio
async def test_mcp_precache():
    mc.configure(MCP_SERVERS=['https://time.mcp.inevitable.fyi/mcp'], VALIDATE_CONFIG=False)
    await mc.env().mcp_registry.precache_tools(connect_timeout=70)
    assert len(mc.mcp_server("time.mcp.inevitable.fyi").get_tools_cache()) >= 1


@pytest.mark.asyncio
async def test_mcp_update_tools_cache():
    mc.configure(MCP_SERVERS=[dict(
        name='test2',
        url='https://time.mcp.inevitable.fyi/mcp'
    )], VALIDATE_CONFIG=False)
    mcp = await mc.mcp.server('test2').connect(fetch_tools=False, connect_timeout=60)
    await mcp.fetch_tools()
    mcp.update_tools_cache()
    await mcp.close()
    assert mcp.tools == mc.mcp.server('test2').get_tools_cache()
    assert len(mcp.tools) >= 1
    ToolsCache.clear()
    assert mc.mcp.server('test2').get_tools_cache() is None


@pytest.mark.asyncio
async def test_mcp_time():
    mc.configure(MCP_SERVERS=['https://time.mcp.inevitable.fyi/mcp'], VALIDATE_CONFIG=False)
    mcp: mc.mcp.MCPConnection = await mc.mcp_server("time.mcp.inevitable.fyi").connect(
        connect_timeout=70
    )
    assert "get_current_time" in mcp.tools
    res = (await mcp.exec(dict(
        call="get_current_time",
        timezone="UTC",
    ))).parse_json()
    assert "datetime" in res
    assert res["timezone"] == "UTC"
    await mcp.close()