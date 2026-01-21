# Querying various LLMs via MCP

This example shows how to create a simple MCP server
that allows you to query different LLMs (OpenAI, Google Gemini, Anthropic Claude)
via a single MCP interface.



## Source Code
[ask_various_llms_mcp_server.py](ask_various_llms_mcp_server.py)

```python
import os, pathlib, fastmcp, dotenv, microcore as mc

dotenv.load_dotenv(pathlib.Path(__file__).parent / '.env', override=True)
mcp = fastmcp.FastMCP("Ask LLMs via MCP", host="0.0.0.0", port=8001)
configs = {  # See https://github.com/Nayjest/ai-microcore?tab=readme-ov-file#%EF%B8%8F-configuring
    "gpt5": {
        "model": "gpt-5",
        "api_type": mc.ApiType.OPENAI,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": "https://api.openai.com/v1",
    },
    "gemini": {
        "model": "gemini-1.5-flash",
        "api_type": mc.ApiType.GOOGLE,
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

@mcp.tool()
def models() -> list[str]: return list(configs.keys())

@mcp.tool()
def ask(query: str, model: str) -> str:
    mc.configure(configs[model])
    return mc.llm(query)

mcp.run(transport="streamable-http")
```

## Usage

**[ 1 ]** Copy ask_various_llms_via_mcp folder containing this file to your computer, navigate inside.

**[ 2 ]** Install dependencies.
```bash
pip install -r requirements.txt
```

> **Note:** `anthropic` and `google-generativeai` packages are optional and required only if you
> want to use Anthropic Claude or Google Gemini models.
> 
> Only `ai-microcore` package is mandatory for usage with OpenAI or local models:
```bash
pip install ai-microcore
```
**[ 3 ]** Copy [.env.template](.env.template) to `.env` and fill in your API keys.

**[ 4 ]** Run the MCP server
```bash
python ask_various_llms_mcp_server.py
```

**[ 5 ]** Connect client to MCP server and test it.

For Claude Desktop use following config: [claude_desktop_config.json](claude_desktop_config.json)

Alternatively you can query MCP server from python code using ai-microcore, see [query_mcp.py](query_mcp.py)

```python
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
            model=model
        )
        print(f"Response from {model}: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Special thanks
Thanks to `AI Talks VLC` community for the idea.
