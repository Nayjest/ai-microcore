# Tool Calls and Finish Reason

A model can be given tools in two ways. MicroCore reads its tool calls the same way in both,
whichever backend is configured and whether the response was streamed or not.

|                        | [Tools described in the prompt](#tools-described-in-the-prompt)     | [Native tool calling](#native-tool-calling)                  |
|------------------------|----------------------------------------------------------------------|--------------------------------------------------------------|
| Tool definitions       | Python functions or MCP tools, rendered into the prompt as text      | `tools=...` passed to `llm()` in the provider's own format   |
| The model calls a tool | In the text of its answer, as JSON                                   | In a separate field of the provider's response               |
| Backends               | Any, including CLI tools and local models                            | OpenAI and compatible APIs, Responses API, Anthropic, Gemini |

In both cases the call is read with the same two methods:

```python
if response.is_tool_call():
    name, args, kwargs = response.as_tool_call()
```

## Tools described in the prompt

### Python functions

```python
from microcore import llm
from microcore.ai_func import ai_func, ToolSet

@ai_func
def get_weather(city: str):
    """Returns the current weather in the city."""
    return f"Sunny in {city}"

toolset = ToolSet([get_weather])
response = llm(
    "What is the weather in Kyiv?\n"
    f"Answer with a call to one of the following tools and nothing else.\n{toolset}"
)
if response.is_tool_call():
    print(toolset.call(*response.as_tool_call()))  # Sunny in Kyiv
```

`{toolset}` renders the description of every function,
see [Python functions as AI tools](../../README.md#python-functions-as-ai-tools).

### MCP servers

```python
import asyncio
from microcore import allm, mcp

async def main():
    server = await mcp.MCPServer("https://mcp.deepwiki.com/mcp").connect()
    response = await allm(
        "What documentation topics exist for the fastapi/fastapi repository?\n"
        f"Answer with a call to one of the following tools and nothing else.\n{server.tools}"
    )
    if response.is_tool_call():
        print(await server.exec(response))

asyncio.run(main())
```

See [MCP](../../README.md#-mcp-model-context-protocol) in the README and the examples:
[one-shot tool call](../../examples/mcp_1shot_deepwiki.py),
[agent loop](../../examples/mcp_agent_loop.py).

## Native tool calling

The tools are defined in the provider's own format and passed to `llm()` as they are:

```python
tools = [{  # OpenAI Chat Completions format
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Returns the current weather in the city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
response = llm("What is the weather in Kyiv?", tools=tools)

for call in response.tool_calls:
    print(call.name, call.args)  # get_weather {'city': 'Kyiv'}
```

The call is run by the same code as before, here with `toolset` from the first example:

```python
if response.is_tool_call():
    print(toolset.call(*response.as_tool_call()))  # Sunny in Kyiv
```

`response.tool_calls` lists every call the model requested, while `as_tool_call()`
returns the first one. Each `ToolCall` has:

| Field       | Type          | Meaning                                                                       |
|-------------|---------------|-------------------------------------------------------------------------------|
| `name`      | `str`         | Name of the function to call                                                  |
| `arguments` | `str`         | Arguments as JSON text, as the model produced them (`"{}"` if there are none) |
| `args`      | `dict`        | `arguments` parsed from JSON (read-only property)                             |
| `id`        | `str \| None` | Identifier the provider expects back with the function result, if it uses one |

An MCP server runs a native call too: `await server.exec(response)` for the first one,
or `await server.exec(call)` for each `call` in `response.tool_calls`.

A response that only calls tools has no text, so it is an empty string.
Text and tool calls may also come together.

### Streaming with callbacks

Callbacks receive text only. Tool calls are assembled from the stream
and are available on the response returned after the stream completes:

```python
response = llm(
    "What is the weather in Kyiv?",
    tools=tools,
    callback=lambda chunk: print(chunk, end=""),
)
print(response.tool_calls)
```

### File cache

Tool calls and the finish reason are stored with the [cached](file_cache.md) response,
so a cache hit returns them as well.

### What native tool calling does not cover

- **Tool definitions.** `tools=...` and related parameters are passed to the provider as they are,
  so they follow the provider's format.
- **Sending tool results back.** Messages with tool calls and tool results
  follow the provider's format as well.
- **Tools executed by the provider** (web search, code execution, etc.) are not tool calls
  for your application and are not listed in `tool_calls`.

The first two do not apply to tools described in the prompt:
their definitions and results are plain text, the same for every backend.

## Finish reason

`response.finish_reason` tells why the generation stopped, in the same terms for every backend:

```python
response = llm("Write a long story")
if response.finish_reason == "length":
    print("The answer was cut off by the output token limit")
```

| Value            | Meaning                                                     |
|------------------|-------------------------------------------------------------|
| `stop`           | The model finished its answer or hit a stop sequence        |
| `length`         | The output token limit or the context window was reached    |
| `tool_calls`     | The model requested native tool calls                       |
| `content_filter` | The answer was withheld or cut by a content / safety filter |
| `None`           | The backend does not report a finish reason                 |

A reason that has no counterpart in this list is returned as the provider reports it
(for example `pause_turn` from Anthropic or `malformed_function_call` from Gemini).

## Provider notes

| Provider                           | Native tool calls                                         | Finish reason source                                                                                                                              |
|------------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| OpenAI and compatible APIs         | `message.tool_calls`, `delta.tool_calls` (function calls) | `finish_reason`                                                                                                                                   |
| Responses API                      | `function_call` output items                              | `status`; `incomplete_details.reason`: `max_output_tokens` → `length`                                                                             |
| Anthropic                          | `tool_use` content blocks                                 | `stop_reason`: `end_turn`, `stop_sequence` → `stop`; `max_tokens` → `length`; `tool_use` → `tool_calls`; `refusal` → `content_filter`             |
| Gemini                             | `function_call` parts                                     | `finish_reason`: `STOP` → `stop`; `MAX_TOKENS` → `length`; `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII` → `content_filter`   |
| CLI, local functions, Transformers | No native tool calling: always empty                      | Not reported: `None`                                                                                                                              |

Gemini, the Responses API and some OpenAI-compatible servers report a regular stop
when the model requests tool calls; MicroCore reports `tool_calls` in that case.

See also: [Token Usage](token_usage.md), the other provider-agnostic part of a response.
