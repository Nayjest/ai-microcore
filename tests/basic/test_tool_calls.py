"""
Native tool calls and finish reason are reported the same way by every backend,
for streamed and non-streamed responses.
"""

import pickle
from unittest.mock import AsyncMock

import pytest
from anthropic.types import (
    CitationCharLocation,
    CitationsDelta,
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    ServerToolUseBlock,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta as MessageDelta
from google.genai import types as genai_types
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response import IncompleteDetails

import microcore as mc
from microcore import ToolCall, ToolDefinition, ToolResult
from microcore.ai_func import ToolSet, ai_func
from microcore.llm.shared import ToolCallAccumulator, resolve_finish_reason

from . import setup  # noqa

WEATHER_CALL = ToolCall("get_weather", '{"city": "Kyiv"}', "call_1")
TIME_CALL = ToolCall("get_time", "{}", "call_2")
WEATHER_TOOL = ToolDefinition(
    "get_weather",
    "Returns the current weather in the city.",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


async def _async_iter(items):
    for item in items:
        yield item


# ---------------------------------------------------------------- building blocks


def test_tool_call_args():
    assert WEATHER_CALL.args == {"city": "Kyiv"}
    assert ToolCall("ping").args == {}
    with pytest.raises(mc.BadAIJsonAnswer):
        _ = ToolCall("ping", '{"broken').args


def test_portable_tool_messages():
    result = ToolResult(WEATHER_CALL, "Sunny in Kyiv")
    response = mc.LLMResponse(
        "Checking.",
        response={"provider": "turn"},
        api_type=mc.ApiType.OPENAI,
        tool_calls=[WEATHER_CALL],
    )

    assert WEATHER_TOOL.name == "get_weather"
    assert WEATHER_TOOL.parameters["required"] == ["city"]
    assert result.role == mc.Role.TOOL
    assert result.call == WEATHER_CALL
    assert result.content == "Sunny in Kyiv"
    message = response.as_message()
    assert message == mc.AssistantMsg(
        content="Checking.",
        tool_calls=[WEATHER_CALL],
        _provider_data={"provider": "turn"},
        _api_type=mc.ApiType.OPENAI,
    )
    assert message._provider_data == {"provider": "turn"}
    assert message._api_type == mc.ApiType.OPENAI


def test_tool_call_accumulator():
    acc = ToolCallAccumulator()
    acc.add(1, call_id="call_2", name="get_time")
    acc.add(0, call_id="call_1", name="get_weather", arguments='{"city"')
    acc.add(0, arguments=': "Kyiv"}')
    assert acc.tool_calls() == [WEATHER_CALL, TIME_CALL]
    assert ToolCallAccumulator().tool_calls() == []


def test_resolve_finish_reason():
    assert resolve_finish_reason("stop", [WEATHER_CALL]) == "tool_calls"
    assert resolve_finish_reason("stop", []) == "stop"
    assert resolve_finish_reason("length", [WEATHER_CALL]) == "length"
    assert resolve_finish_reason(None, [WEATHER_CALL]) is None


def test_llm_response_defaults():
    response = mc.LLMResponse("Hi")
    assert response.tool_calls == []
    assert response.finish_reason is None
    assert mc.LLMResponse("Hi").tool_calls is not response.tool_calls


def test_tool_calls_survive_file_cache_serialization():
    response = mc.LLMResponse("", tool_calls=[WEATHER_CALL], finish_reason="tool_calls")
    restored = pickle.loads(pickle.dumps(response))
    assert restored.tool_calls == [WEATHER_CALL]
    assert restored.finish_reason == "tool_calls"


def test_backend_without_native_tool_calling(setup):
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.FUNCTION,
        INFERENCE_FUNC=lambda prompt, **kwargs: "Hi",
    )
    response = mc.llm("Hello")
    assert response == "Hi"
    assert response.tool_calls == []
    assert response.finish_reason is None


# ---------------------------------------------------------------- OpenAI Chat Completions


def _oai_completion(
    content=None, tool_calls=(), finish_reason="stop"
) -> ChatCompletion:
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=call.id,
                type="function",
                function=Function(name=call.name, arguments=call.arguments),
            )
            for call in tool_calls
        ]
        or None,
    )
    return ChatCompletion(
        id="1",
        created=1,
        model="m",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason=finish_reason, message=message)],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _oai_chunk(finish_reason=None, usage=None, **delta) -> ChatCompletionChunk:
    choices = [
        ChunkChoice(index=0, delta=ChoiceDelta(**delta), finish_reason=finish_reason)
    ]
    return ChatCompletionChunk(
        id="1",
        created=1,
        model="m",
        object="chat.completion.chunk",
        choices=[] if usage else choices,
        usage=usage,
    )


def _oai_call_delta(
    index, call_id=None, name=None, arguments=None
) -> ChoiceDeltaToolCall:
    return ChoiceDeltaToolCall(
        index=index,
        id=call_id,
        type="function" if call_id else None,
        function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
    )


def _oai_stream(
    text_chunks=(), finish_reason="tool_calls"
) -> list[ChatCompletionChunk]:
    """Optional text, then two parallel tool calls with fragmented arguments."""
    return [
        _oai_chunk(role="assistant"),
        *[_oai_chunk(content=text) for text in text_chunks],
        _oai_chunk(tool_calls=[_oai_call_delta(0, "call_1", "get_weather", '{"city"')]),
        _oai_chunk(tool_calls=[_oai_call_delta(1, "call_2", "get_time", "")]),
        _oai_chunk(tool_calls=[_oai_call_delta(0, arguments=': "Kyiv"}')]),
        _oai_chunk(finish_reason=finish_reason),
        _oai_chunk(
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
        ),
    ]


def _configure_openai_hiding_output():
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.OPENAI,
        LLM_API_KEY="123",
        HIDDEN_OUTPUT_BEGIN="<hide>",
        HIDDEN_OUTPUT_END="</hide>",
    )


def test_openai(setup, mocker):
    mocker.patch(
        "openai.resources.chat.Completions.create",
        return_value=_oai_completion(
            tool_calls=[WEATHER_CALL], finish_reason="tool_calls"
        ),
    )
    response = mc.llm("What is the weather in Kyiv?", tools=[])
    assert response == ""
    assert response.tool_calls == [WEATHER_CALL]
    assert response.finish_reason == "tool_calls"
    assert response.usage["total_tokens"] == 15


def test_openai_portable_tool_round_trip(setup, mocker):
    create = mocker.patch(
        "openai.resources.chat.Completions.create",
        side_effect=[
            _oai_completion(tool_calls=[WEATHER_CALL], finish_reason="tool_calls"),
            _oai_completion(content="It is sunny."),
        ],
    )
    prompt = mc.UserMsg("What is the weather in Kyiv?")
    response = mc.llm(prompt, tools=[WEATHER_TOOL])
    mc.llm(
        [prompt, response.as_message(), ToolResult(WEATHER_CALL, "Sunny")],
        tools=[WEATHER_TOOL],
    )

    expected_tool = {
        "type": "function",
        "function": {
            "name": WEATHER_TOOL.name,
            "description": WEATHER_TOOL.description,
            "parameters": WEATHER_TOOL.parameters,
        },
    }
    assert create.call_args_list[0].kwargs["tools"] == [expected_tool]
    assert create.call_args_list[1].kwargs["messages"] == [
        {"role": mc.Role.USER, "content": "What is the weather in Kyiv?"},
        {
            "role": mc.Role.ASSISTANT,
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Kyiv"}',
                    },
                }
            ],
        },
        {"role": mc.Role.TOOL, "tool_call_id": "call_1", "content": "Sunny"},
    ]


async def test_openai_async(setup, mocker):
    mocker.patch(
        "openai.resources.chat.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=_oai_completion(
            tool_calls=[WEATHER_CALL], finish_reason="tool_calls"
        ),
    )
    response = await mc.allm("What is the weather in Kyiv?", tools=[])
    assert response == ""
    assert response.tool_calls == [WEATHER_CALL]
    assert response.finish_reason == "tool_calls"


def test_openai_text_response(setup, mocker):
    mocker.patch(
        "openai.resources.chat.Completions.create",
        return_value=_oai_completion(content="Trunca", finish_reason="length"),
    )
    response = mc.llm("Hello")
    assert response == "Trunca"
    assert response.tool_calls == []
    assert response.finish_reason == "length"


def test_openai_hidden_output_with_tool_call_only(setup, mocker):
    mocker.patch(
        "openai.resources.chat.Completions.create",
        return_value=_oai_completion(
            tool_calls=[WEATHER_CALL], finish_reason="tool_calls"
        ),
    )
    _configure_openai_hiding_output()
    assert mc.llm("What is the weather in Kyiv?", tools=[]) == ""


def test_openai_stream(setup, mocker):
    mocker.patch(
        "openai.resources.chat.Completions.create",
        return_value=iter(_oai_stream(text_chunks=["Let me ", "check."])),
    )
    chunks = []
    response = mc.llm("What is the weather in Kyiv?", tools=[], callback=chunks.append)
    assert chunks == ["Let me ", "check."]
    assert response == "Let me check."
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]
    assert response.finish_reason == "tool_calls"
    assert response.usage["total_tokens"] == 15


async def test_openai_stream_async(setup, mocker):
    mocker.patch(
        "openai.resources.chat.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=_async_iter(_oai_stream()),
    )
    chunks = []

    async def callback(chunk):
        chunks.append(chunk)

    response = await mc.allm(
        "What is the weather in Kyiv?", tools=[], callback=callback
    )
    assert chunks == []
    assert response == ""
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]
    assert response.finish_reason == "tool_calls"


def test_openai_stream_reports_tool_calls_when_server_says_stop(setup, mocker):
    mocker.patch(
        "openai.resources.chat.Completions.create",
        return_value=iter(_oai_stream(finish_reason="stop")),
    )
    response = mc.llm("What is the weather in Kyiv?", tools=[], stream=True)
    assert response.finish_reason == "tool_calls"


def test_openai_stream_hidden_output(setup, mocker):
    mocker.patch(
        "openai.resources.chat.Completions.create",
        return_value=iter(
            _oai_stream(text_chunks=["<hide>", "secret", "</hide>", "visible"])
        ),
    )
    _configure_openai_hiding_output()
    chunks = []
    response = mc.llm("What is the weather in Kyiv?", tools=[], callback=chunks.append)
    assert chunks == ["visible"]
    assert response == "visible"
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]


# ---------------------------------------------------------------- OpenAI Responses API


def _function_call_item(call: ToolCall, arguments=None) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        id="fc_" + call.id,
        call_id=call.id,
        name=call.name,
        arguments=call.arguments if arguments is None else arguments,
    )


def _responses_stream(final_event) -> list:
    return [
        ResponseTextDeltaEvent(
            type="response.output_text.delta",
            content_index=0,
            delta="Checking.",
            item_id="msg_1",
            logprobs=[],
            output_index=0,
            sequence_number=1,
        ),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=_function_call_item(WEATHER_CALL, arguments=""),
            output_index=1,
            sequence_number=2,
        ),
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            delta='{"city"',
            item_id="fc_call_1",
            output_index=1,
            sequence_number=3,
        ),
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            delta=': "Kyiv"}',
            item_id="fc_call_1",
            output_index=1,
            sequence_number=4,
        ),
        final_event,
    ]


def test_openai_responses_api(setup, mocker):
    mocker.patch(
        "openai.resources.responses.Responses.create",
        return_value=Response.model_construct(
            output=[_function_call_item(WEATHER_CALL)], status="completed", usage=None
        ),
    )
    response = mc.llm("What is the weather in Kyiv?", tools=[], use_responses_api=True)
    assert response == ""
    assert response.tool_calls == [WEATHER_CALL]
    assert response.finish_reason == "tool_calls"


def test_openai_responses_portable_tool_round_trip(setup, mocker):
    first = Response.model_construct(
        output=[_function_call_item(WEATHER_CALL)], status="completed", usage=None
    )
    create = mocker.patch(
        "openai.resources.responses.Responses.create",
        side_effect=[
            first,
            Response.model_construct(output=[], status="completed", usage=None),
        ],
    )
    prompt = mc.UserMsg("What is the weather in Kyiv?")
    response = mc.llm(prompt, tools=[WEATHER_TOOL], use_responses_api=True)
    mc.llm(
        [prompt, response.as_message(), ToolResult(WEATHER_CALL, "Sunny")],
        tools=[WEATHER_TOOL],
        use_responses_api=True,
    )

    assert create.call_args_list[0].kwargs["tools"] == [
        {
            "type": "function",
            "name": WEATHER_TOOL.name,
            "description": WEATHER_TOOL.description,
            "parameters": WEATHER_TOOL.parameters,
        }
    ]
    second_input = create.call_args_list[1].kwargs["input"]
    assert second_input[0] == {
        "type": "message",
        "role": mc.Role.USER,
        "content": "What is the weather in Kyiv?",
    }
    assert second_input[1]["type"] == "function_call"
    assert second_input[1]["call_id"] == "call_1"
    assert second_input[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "Sunny",
    }


def test_openai_responses_api_stream(setup, mocker):
    completed = ResponseCompletedEvent.model_construct(
        type="response.completed",
        response=Response.model_construct(status="completed", usage=None),
    )
    mocker.patch(
        "openai.resources.responses.Responses.create",
        return_value=iter(_responses_stream(completed)),
    )
    chunks = []
    response = mc.llm(
        "What is the weather in Kyiv?",
        tools=[],
        use_responses_api=True,
        callback=chunks.append,
    )
    assert chunks == ["Checking."]
    assert response == "Checking."
    assert response.tool_calls == [WEATHER_CALL]
    assert response.finish_reason == "tool_calls"


async def test_openai_responses_api_stream_async_incomplete(setup, mocker):
    incomplete = ResponseIncompleteEvent.model_construct(
        type="response.incomplete",
        response=Response.model_construct(
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
            usage=None,
        ),
    )
    mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
        return_value=_async_iter(_responses_stream(incomplete)),
    )
    response = await mc.allm(
        "What is the weather in Kyiv?", tools=[], use_responses_api=True, stream=True
    )
    assert response.tool_calls == [WEATHER_CALL]
    assert response.finish_reason == "length"


# ---------------------------------------------------------------- Anthropic


@pytest.fixture()
def anthropic_config(setup):
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.ANTHROPIC,
        LLM_API_KEY="123",
        MODEL="claude-sonnet-4-6",
    )


def _anthropic_message(content=(), stop_reason=None) -> Message:
    return Message(
        id="1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=list(content),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _anthropic_block(index, block, deltas) -> list:
    return [
        RawContentBlockStartEvent(
            type="content_block_start", index=index, content_block=block
        ),
        *[
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=index, delta=delta
            )
            for delta in deltas
        ],
        RawContentBlockStopEvent(type="content_block_stop", index=index),
    ]


def _input_json(partial_json) -> InputJSONDelta:
    return InputJSONDelta(type="input_json_delta", partial_json=partial_json)


def _anthropic_stream(stop_reason="tool_use") -> list:
    """Text with a citation, a server tool, then two tool calls (the second without input)."""
    citation = CitationCharLocation(
        type="char_location",
        cited_text="x",
        document_index=0,
        document_title=None,
        start_char_index=0,
        end_char_index=1,
    )
    return [
        RawMessageStartEvent(type="message_start", message=_anthropic_message()),
        *_anthropic_block(
            0,
            TextBlock(type="text", text=""),
            [
                TextDelta(type="text_delta", text="Checking."),
                CitationsDelta(type="citations_delta", citation=citation),
            ],
        ),
        *_anthropic_block(
            1,
            ServerToolUseBlock(
                type="server_tool_use", id="srv_1", name="web_search", input={}
            ),
            [_input_json('{"query": "Kyiv"}')],
        ),
        *_anthropic_block(
            2,
            ToolUseBlock(type="tool_use", id="call_1", name="get_weather", input={}),
            [_input_json('{"city"'), _input_json(': "Kyiv"}')],
        ),
        *_anthropic_block(
            3,
            ToolUseBlock(type="tool_use", id="call_2", name="get_time", input={}),
            [_input_json("")],
        ),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(stop_reason=stop_reason),
            usage=MessageDeltaUsage(output_tokens=20),
        ),
    ]


def test_anthropic(anthropic_config, mocker):
    mocker.patch(
        "anthropic.resources.Messages.create",
        return_value=_anthropic_message(
            [
                TextBlock(type="text", text="Checking."),
                ToolUseBlock(
                    type="tool_use",
                    id="call_1",
                    name="get_weather",
                    input={"city": "Kyiv"},
                ),
                ToolUseBlock(type="tool_use", id="call_2", name="get_time", input={}),
            ],
            stop_reason="tool_use",
        ),
    )
    response = mc.llm("What is the weather in Kyiv?", tools=[])
    assert response == "Checking."
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]
    assert response.finish_reason == "tool_calls"


def test_anthropic_portable_tool_round_trip(anthropic_config, mocker):
    create = mocker.patch(
        "anthropic.resources.Messages.create",
        side_effect=[
            _anthropic_message(
                [
                    ToolUseBlock(
                        type="tool_use",
                        id="call_1",
                        name="get_weather",
                        input={"city": "Kyiv"},
                    ),
                    ToolUseBlock(
                        type="tool_use", id="call_2", name="get_time", input={}
                    ),
                ],
                stop_reason="tool_use",
            ),
            _anthropic_message([TextBlock(type="text", text="It is sunny.")]),
        ],
    )
    prompt = mc.UserMsg("What is the weather in Kyiv?")
    response = mc.llm(prompt, tools=[WEATHER_TOOL])
    mc.llm(
        [
            prompt,
            response.as_message(),
            ToolResult(WEATHER_CALL, "Sunny"),
            ToolResult(TIME_CALL, "10:00"),
        ],
        tools=[WEATHER_TOOL],
    )

    assert create.call_args_list[0].kwargs["tools"] == [
        {
            "name": WEATHER_TOOL.name,
            "description": WEATHER_TOOL.description,
            "input_schema": WEATHER_TOOL.parameters,
        }
    ]
    messages = create.call_args_list[1].kwargs["messages"]
    assert messages[1]["role"] == mc.Role.ASSISTANT
    assert messages[1]["content"][0].type == "tool_use"
    assert messages[2] == {
        "role": mc.Role.USER,
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "Sunny",
            },
            {
                "type": "tool_result",
                "tool_use_id": "call_2",
                "content": "10:00",
            },
        ],
    }


@pytest.mark.parametrize(
    "stop_reason, finish_reason",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("pause_turn", "pause_turn"),
        (None, None),
    ],
)
async def test_anthropic_async_finish_reason(
    anthropic_config, mocker, stop_reason, finish_reason
):
    mocker.patch(
        "anthropic.resources.AsyncMessages.create",
        new_callable=AsyncMock,
        return_value=_anthropic_message(
            [TextBlock(type="text", text="Hi")], stop_reason=stop_reason
        ),
    )
    response = await mc.allm("Hello")
    assert response == "Hi"
    assert response.tool_calls == []
    assert response.finish_reason == finish_reason


def test_anthropic_stream(anthropic_config, mocker):
    mocker.patch(
        "anthropic.resources.Messages.create", return_value=iter(_anthropic_stream())
    )
    chunks = []
    response = mc.llm("What is the weather in Kyiv?", tools=[], callback=chunks.append)
    assert chunks == ["Checking."]
    assert response == "Checking."
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]
    assert response.finish_reason == "tool_calls"
    assert response.usage["completion_tokens"] == 20


async def test_anthropic_stream_async(anthropic_config, mocker):
    mocker.patch(
        "anthropic.resources.AsyncMessages.create",
        new_callable=AsyncMock,
        return_value=_async_iter(_anthropic_stream(stop_reason="max_tokens")),
    )
    response = await mc.allm("What is the weather in Kyiv?", tools=[], stream=True)
    assert response == "Checking."
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]
    assert response.finish_reason == "length"


# ---------------------------------------------------------------- Google GenAI


@pytest.fixture()
def google_config(setup):
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.GOOGLE,
        LLM_API_PLATFORM=mc.ApiPlatform.GOOGLE_AI_STUDIO,
        LLM_API_KEY="123",
        MODEL="gemini-2.5-flash",
    )


def _google_response(
    text=None, tool_calls=(), finish_reason="STOP"
) -> genai_types.GenerateContentResponse:
    parts = [genai_types.Part(text=text)] if text else []
    parts += [
        genai_types.Part(
            function_call=genai_types.FunctionCall(
                id=call.id, name=call.name, args=call.args
            )
        )
        for call in tool_calls
    ]
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=parts),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        ),
    )


def _google_stream() -> list:
    return [
        _google_response(text="Checking.", finish_reason=None),
        _google_response(tool_calls=[WEATHER_CALL], finish_reason=None),
        _google_response(tool_calls=[TIME_CALL]),
    ]


def test_google(google_config, mocker):
    mocker.patch(
        "google.genai.models.Models.generate_content",
        return_value=_google_response(tool_calls=[WEATHER_CALL, TIME_CALL]),
    )
    response = mc.llm("What is the weather in Kyiv?")
    assert response == ""
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]
    assert response.finish_reason == "tool_calls"
    assert response.usage["total_tokens"] == 15


def test_google_portable_tool_round_trip(google_config, mocker):
    create = mocker.patch(
        "google.genai.models.Models.generate_content",
        side_effect=[
            _google_response(tool_calls=[WEATHER_CALL]),
            _google_response(text="It is sunny."),
        ],
    )
    prompt = mc.UserMsg("What is the weather in Kyiv?")
    response = mc.llm(prompt, tools=[WEATHER_TOOL])
    mc.llm(
        [prompt, response.as_message(), ToolResult(WEATHER_CALL, "Sunny")],
        tools=[WEATHER_TOOL],
    )

    config = create.call_args_list[0].kwargs["config"]
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == WEATHER_TOOL.name
    assert declaration.description == WEATHER_TOOL.description
    assert declaration.parameters.required == ["city"]
    contents = create.call_args_list[1].kwargs["contents"]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0].function_call.name == "get_weather"
    function_response = contents[2]["parts"][0].function_response
    assert function_response.id == "call_1"
    assert function_response.name == "get_weather"
    assert function_response.response == {"output": "Sunny"}


@pytest.mark.parametrize(
    "google_reason, finish_reason",
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "malformed_function_call"),
        ("FINISH_REASON_UNSPECIFIED", None),
        (None, None),
    ],
)
async def test_google_async_finish_reason(
    google_config, mocker, google_reason, finish_reason
):
    mocker.patch(
        "google.genai.models.AsyncModels.generate_content",
        new_callable=AsyncMock,
        return_value=_google_response(text="Hi", finish_reason=google_reason),
    )
    response = await mc.allm("Hello")
    assert response == "Hi"
    assert response.tool_calls == []
    assert response.finish_reason == finish_reason


def test_google_stream(google_config, mocker):
    mocker.patch(
        "google.genai.models.Models.generate_content_stream",
        return_value=iter(_google_stream()),
    )
    chunks = []
    response = mc.llm("What is the weather in Kyiv?", callback=chunks.append)
    assert chunks == ["Checking."]
    assert response == "Checking."
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]
    assert response.finish_reason == "tool_calls"
    assert response.usage["total_tokens"] == 15


async def test_google_stream_async(google_config, mocker):
    mocker.patch(
        "google.genai.models.AsyncModels.generate_content_stream",
        new_callable=AsyncMock,
        return_value=_async_iter(_google_stream()),
    )
    response = await mc.allm("What is the weather in Kyiv?", stream=True)
    assert response == "Checking."
    assert response.tool_calls == [WEATHER_CALL, TIME_CALL]
    assert response.finish_reason == "tool_calls"


# ---------------------------------------------------------------- one way to read both kinds


def get_weather(city: str):
    """Returns the weather in the city."""
    return f"Sunny in {city}"


def test_native_tool_call_is_a_tool_call(setup):
    response = mc.LLMResponse("", tool_calls=[WEATHER_CALL, TIME_CALL])
    assert response.is_tool_call()
    assert response.as_tool_call() == ("get_weather", [], {"city": "Kyiv"})

    toolset = ToolSet([ai_func(get_weather)], verbose=False)
    assert toolset.call(*response.as_tool_call(toolset)) == "Sunny in Kyiv"


def test_tool_call_in_text_is_read_as_before(setup):
    response = mc.LLMResponse('{"call": "get_weather", "city": "Kyiv"}')
    assert response.tool_calls == []
    assert response.is_tool_call()
    assert response.as_tool_call() == ("get_weather", [], {"city": "Kyiv"})

    toolset = ToolSet([ai_func(get_weather)], verbose=False)
    assert toolset.call(*response.as_tool_call(toolset)) == "Sunny in Kyiv"


def test_plain_answer_is_not_a_tool_call(setup):
    response = mc.LLMResponse("It is sunny.")
    assert not response.is_tool_call()
    assert response.as_tool_call() is None


def test_native_tool_call_comes_before_the_one_in_text(setup):
    response = mc.LLMResponse('{"call": "get_time"}', tool_calls=[WEATHER_CALL])
    assert response.as_tool_call() == ("get_weather", [], {"city": "Kyiv"})


def test_native_tool_call_does_not_parse_the_text(setup):
    """Parsing the text of a response made with `retries` may repeat the LLM request."""
    response = mc.LLMResponse("Let me check.", tool_calls=[WEATHER_CALL])
    response._retry_callback = lambda: pytest.fail("The LLM request was repeated")
    assert response.is_tool_call()
    assert response.as_tool_call() == ("get_weather", [], {"city": "Kyiv"})


def test_response_cached_by_an_older_version(setup):
    """Responses pickled before `tool_calls` and `finish_reason` existed get the defaults."""
    response = mc.LLMResponse('{"call": "get_weather", "city": "Kyiv"}')
    del response.tool_calls, response.finish_reason
    restored = pickle.loads(pickle.dumps(response))
    assert restored.tool_calls == []
    assert restored.finish_reason is None
    assert restored.as_tool_call() == ("get_weather", [], {"city": "Kyiv"})
