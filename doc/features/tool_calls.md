# Tool Calls and Finish Reason

When a model answers with native tool (function) calls, every LLM response carries them
in the same provider-agnostic form, together with the reason the generation stopped.
Code that reads them stays the same whichever backend is configured,
and whether the response was streamed or not.

## Two ways to give tools to a model

|                        | Tools described in the prompt                              | Native tool calling                                       |
|------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| How                    | `@ai_func`, `ToolSet` or MCP tools rendered into the prompt | `tools=...` passed to `llm()` in the provider's format     |
| The model answers      | In the text of the response (JSON or a tag)                | In a separate field of the provider's response            |
| Backends               | Any, including CLI and local models                        | OpenAI and compatible APIs, Responses API, Anthropic, Gemini |
| `response.tool_calls`  | Empty                                                      | The calls                                                 |

This page is about the second way. Whichever way was used,
the same methods tell whether the model called a tool and run it:

```python
if response.is_tool_call():
    name, args, kwargs = response.as_tool_call()
    result = toolset.call(name, args, kwargs)  # or: await mcp.exec(response)
```

For a natively called tool, `as_tool_call()` and `mcp.exec(response)` use the first of
`response.tool_calls`; otherwise they read the call from the text.
To run each of several native calls, pass them one by one:

```python
for call in response.tool_calls:
    result = await mcp.exec(call)
```

## Fields

| Field           | Type             | Meaning                                                            |
|-----------------|------------------|--------------------------------------------------------------------|
| `tool_calls`    | `list[ToolCall]` | Function calls requested by the model; empty if there are none     |
| `finish_reason` | `str \| None`    | Why the generation stopped, see [Finish reason](#finish-reason)    |

`ToolCall` fields:

| Field       | Type          | Meaning                                                                    |
|-------------|---------------|----------------------------------------------------------------------------|
| `name`      | `str`         | Name of the function to call                                               |
| `arguments` | `str`         | Arguments as JSON text, as the model produced them (`"{}"` if there are none) |
| `id`        | `str \| None` | Identifier the provider expects back with the function result, if it uses one |
| `args`      | `dict`        | `arguments` parsed from JSON (read-only property)                          |

## Reading tool calls

```python
from microcore import llm

response = llm("What is the weather in Kyiv?", tools=tools)

for call in response.tool_calls:
    print(call.name, call.args)  # get_weather {'city': 'Kyiv'}

if response.finish_reason == "length":
    print("The answer was cut off by the output token limit")
```

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
print(response.tool_calls, response.finish_reason)
```

### File cache

Tool calls and the finish reason are stored with the cached response,
so a cache hit returns them as well.

## Finish reason

| Value            | Meaning                                                    |
|------------------|------------------------------------------------------------|
| `stop`           | The model finished its answer or hit a stop sequence       |
| `length`         | The output token limit or the context window was reached   |
| `tool_calls`     | The model requested tool calls                             |
| `content_filter` | The answer was withheld or cut by a content / safety filter |
| `None`           | The backend does not report a finish reason                |

A reason that has no counterpart in this list is returned as the provider reports it
(for example `pause_turn` from Anthropic or `malformed_function_call` from Gemini).

## What is not covered

- **Tool definitions.** `tools=...` and related parameters are passed to the provider as they are,
  in the provider's own format.
- **Sending tool results back.** Messages with tool calls and tool results
  are passed in the provider's own format as well.
- **Tools executed by the provider** (web search, code execution, etc.) are not tool calls
  for your application and are not listed in `tool_calls`.

Tools described in the prompt (`@ai_func`, MCP) have neither limitation:
their definitions and results are plain text, the same for every backend.

## Provider notes

| Provider                    | Tool calls                                         | Finish reason source                                                                 |
|-----------------------------|----------------------------------------------------|--------------------------------------------------------------------------------------|
| OpenAI and compatible APIs  | `message.tool_calls`, `delta.tool_calls` (function calls) | `finish_reason`                                                               |
| Responses API               | `function_call` output items                       | `status`; `incomplete_details.reason`: `max_output_tokens` → `length`                |
| Anthropic                   | `tool_use` content blocks                          | `stop_reason`: `end_turn`, `stop_sequence` → `stop`; `max_tokens` → `length`; `tool_use` → `tool_calls`; `refusal` → `content_filter` |
| Gemini                      | `function_call` parts                              | `finish_reason`: `STOP` → `stop`; `MAX_TOKENS` → `length`; `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII` → `content_filter` |
| CLI, local functions, Transformers | No native tool calling: always empty        | Not reported: `None`                                                                 |

Gemini, the Responses API and some OpenAI-compatible servers report a regular stop
when the model requests tool calls; MicroCore reports `tool_calls` in that case.
