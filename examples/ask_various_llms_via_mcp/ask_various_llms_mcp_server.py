import os, pathlib, fastmcp, dotenv, microcore as mc

dotenv.load_dotenv(pathlib.Path(__file__).parent / '.env', override=True)
configs = {  # See https://github.com/Nayjest/ai-microcore?tab=readme-ov-file#%EF%B8%8F-configuring
    "gpt-5": {
        "model": "gpt-5",
        "api_type": mc.ApiType.OPENAI,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": "https://api.openai.com/v1",
    },
    "grok-4": {
        "model": "grok-4-latest",
        "api_type": mc.ApiType.OPENAI,
        "api_key": os.getenv("XAI_API_KEY"),
        "api_base": "https://api.x.ai/v1",
    },
    "deepseek-chat": {
        "model": "deepseek-chat",
        "api_type": mc.ApiType.OPENAI,
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "api_base": "https://api.deepseek.com/v1",
    },
    "gemini-2.5-pro": {
        "model": "gemini-2.5-pro",
        "api_type": mc.ApiType.GOOGLE,
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "api_base": "https://generativelanguage.googleapis.com/v1alpha",
    },
    "claude-opus-4.1": {
        "model": "claude-opus-4-1-20250805",
        "api_type": mc.ApiType.ANTHROPIC,
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "api_base": "",
    },
}

mcp = fastmcp.FastMCP("Ask LLMs via MCP")


@mcp.tool()
def models() -> list[str]: return list(configs.keys())


@mcp.tool()
def ask(query: str, model: str) -> str:
    mc.configure(configs[model])
    return mc.llm(query)


mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
