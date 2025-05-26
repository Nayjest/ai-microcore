import asyncio

import microcore as mc

mc.configure(
    interactive_setup=True,
    dot_env_file='~/.env.ai-code-review',
    use_logging=True,
    mcp_servers=[
        {"name": "fetch", "url": 'https://remote.mcpservers.org/fetch/mcp', }
    ]
)


async def main():
    mcp = await mc.mcp.server('fetch').connect()
    prompt = mc.prompt("""
        How many stars nayjest/ai-microcore has on GitHub?
        Use tools to answer the question.
        {{ tools }}
        """, tools=mcp.tools
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
