import asyncio

from microcore import allm, mcp, configure, ui

configure(DOT_ENV_FILE="~/.ai-microcore.env", INTERACTIVE_SETUP=True)


async def main():
    question = "What documentation topics exist for the fastapi/fastapi repository?"
    # DeepWiki: public MCP server answering questions about GitHub repositories
    mcp_conn = await mcp.MCPServer("https://mcp.deepwiki.com/mcp").connect()
    tool_call = await allm(
        f"{question}\n"
        f"Answer with a call to one of the following tools and nothing else.\n{mcp_conn.tools}"
    )
    data = await mcp_conn.exec(tool_call)
    answer = await allm([question, data])
    print(f"The answer is {ui.green(answer)}")


if __name__ == "__main__":
    asyncio.run(main())
