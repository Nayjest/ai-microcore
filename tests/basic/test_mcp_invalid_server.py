import logging
import os
import microcore as mc
import pytest

servers_cfg = [
    {"name": "mcp1", "url": "http://localhost:8899"},
]

@pytest.mark.asyncio
async def test_bad_mcp():
    mc.configure(
        LLM_API_TYPE=mc.ApiType.NONE,
        MCP_SERVERS=servers_cfg,
    )
    await mc.env().mcp_registry.precache_tools(connect_timeout=0.1)
    with pytest.raises(TimeoutError) as exc_info:
        await mc.env().mcp_registry.precache_tools(raise_errors=True, connect_timeout=0.1)
    logging.info("OK")