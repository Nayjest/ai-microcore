import asyncio
import microcore as mc


async def main():
    mc.configure(LLM_API_TYPE=mc.ApiType.NONE, MCP_SERVERS=['http://localhost:8001'])
    mcp = await mc.mcp.server('localhost:8001').connect()
    models = (await mcp.call('models')).parse_json()
    print('Models:', models)
    for model in models:
        response = await mcp.call(
            'ask',
            query="""
            What is your favorite video game character name?
            Just give me the name, no explanation or details.""",
            model=model,
            timeout=300,
        )
        print(f"Response from {model}: {response}")


if __name__ == "__main__":
    asyncio.run(main())
