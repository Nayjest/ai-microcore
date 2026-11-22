# Tool Calls and Finish Reason

When a model answers with native tool (function) calls, every LLM response carries them
in the same provider-agnostic form, together with the reason the generation stopped.
Code that reads them stays the same whichever backend is configured,
and whether the response was streamed or not.

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

For provider-independent tool usage that also works with models
without native tool calling, see `@ai_func` and the MCP integration.

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
