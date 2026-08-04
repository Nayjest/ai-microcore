import asyncio

import microcore as mc

mc.configure(
    interactive_setup=True,
    dot_env_file="~/.env.ai-code-review",
    use_logging=True,
    mcp_servers=[
        # DeepWiki: public MCP server answering questions about GitHub repositories
        {
            "name": "deepwiki",
            "url": "https://mcp.deepwiki.com/mcp",
        }
    ],
)


async def main():
    mcp = await mc.mcp.server("deepwiki").connect()
    prompt = mc.prompt(
        """
        What documentation topics exist for the fastapi/fastapi repository?
        To use a tool, respond with the corresponding JSON and nothing else.
        Available tools:
        {{ tools }}
        When you have the tool results, answer in plain text.
        """,
        tools=mcp.tools,
    )
    chat = [prompt]
    while True:
        llm_response = await mc.allm(chat)
        if not llm_response.is_tool_call():
            break
        mcp_result = await mcp.exec(llm_response)
        chat += [llm_response.as_assistant, mcp_result.as_assistant]
    print("Answer:", llm_response)


if __name__ == "__main__":
    asyncio.run(main())
