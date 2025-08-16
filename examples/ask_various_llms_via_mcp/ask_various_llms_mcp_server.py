import os, pathlib, fastmcp, dotenv, microcore as mc

dotenv.load_dotenv(pathlib.Path(__file__).parent / '.env', override=True)
configs = {
    "gpt5": {
        "model": "gpt-5",
        "api_type": mc.ApiType.OPEN_AI,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": "https://api.openai.com/v1",
    },
    "gemini": {
        "model": "gemini-1.5-flash",
        "api_type": mc.ApiType.GOOGLE_AI_STUDIO,
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "api_base": "https://generativelanguage.googleapis.com/v1alpha",
    },
    "claude": {
        "model": "claude-opus-4-1-20250805",
        "api_type": mc.ApiType.ANTHROPIC,
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "api_base": "",
    },
}

mcp = fastmcp.FastMCP("Ask LLMs via MCP", host="0.0.0.0", port=8001)


@mcp.tool()
def models() -> list[str]: return list(configs.keys())

@mcp.tool()
def ask(query: str, model: str) -> str:
    mc.configure(configs[model])
    return mc.llm(query)


mcp.run(transport="streamable-http")
