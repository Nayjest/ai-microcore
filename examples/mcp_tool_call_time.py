import asyncio

import microcore as mc

mc.configure(MCP_SERVERS=['https://time.mcp.inevitable.fyi/mcp'], VALIDATE_CONFIG=False)

async def main():
    mcp = await mc.mcp_server("time.mcp.inevitable.fyi").connect()
    print(mc.ui.magenta("Tools:\n"), mcp.tools)
    print(mc.ui.green("Current time from MCP:"), await mcp.exec(dict(
        call="get_current_time",
        timezone="UTC",
    )))
    await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())
