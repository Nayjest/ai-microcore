import logging
import pytest
from pathlib import Path
import subprocess
import microcore as mc
from microcore.mcp import ToolsCache
from time import sleep


TRANSPORTS = [
    "sse",
    # "stdio", # not supported
    "streamable-http"
]


@pytest.fixture(scope="session", params=TRANSPORTS)
def server(request):
    process = None
    try:
        port = 5000 + TRANSPORTS.index(request.param)  # Unique port per transport
        executable = "python"
        cmd = [executable, "ping_server.py", "--port", str(port), "--transport", request.param]
        logging.info(
            f"Starting MCP server with transport: "
            f"{request.param} on port {port}: {mc.ui.yellow(' '.join(cmd))}"
        )
        process = None
        process = subprocess.Popen(
            cmd,
            cwd=(Path(__file__).parent / "mcp_servers").resolve(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        sleep(3)
        yield {"process": process, "port": port, "transport": request.param}
    finally:
        if process:
            process.terminate()
            process.wait()
            stdout, stderr = process.communicate()
            if stdout:
                logging.info(f"Server stdout: {stdout.decode()}")
            if stderr:
                logging.error(f"Server stderr: {stderr.decode()}")


@pytest.mark.asyncio
async def test_mcp_ping(server):
    logging.info(
        f"Testing MCP server with transport: {server['transport']} on port {server['port']}"
    )
    mc.configure(
        LLM_API_TYPE=mc.ApiType.NONE,
        MCP_SERVERS=[{
            "name": "test_mcp",
            "url": f"http://127.0.0.1:{server['port']}",
        }],
    )
    ToolsCache.clear()
    mcp = await mc.mcp_server("test_mcp").connect(connect_timeout=10)
    logging.info("Ping...")
    assert await mcp.call("ping", message="1") == "pong 1"
    assert await mcp.exec(dict(call="ping", message="2")) == "pong 2"
    await mcp.close()
