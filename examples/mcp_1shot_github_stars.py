import asyncio

from microcore import allm, mcp, configure, ui
configure(DOT_ENV_FILE="~/.ai-microcore.env", INTERACTIVE_SETUP=True)


async def main():
    question = "How many stars nayjest/ai-microcore has on GitHub?"
    # This is public MCP capable of fetching web-pages
    mcp_conn = await mcp.MCPServer("https://remote.mcpservers.org/fetch/mcp").connect()
    tool_params = await allm(f"{question}\nUse tools to answer the question.\n{mcp_conn.tools}")
    data = await mcp_conn.exec(tool_params)
    answer = await allm([question, data])
    print(f"The answer in {ui.green(answer.parse_number(dtype=int))}")


if __name__ == "__main__":
    asyncio.run(main())
