from types import SimpleNamespace

import pytest
from anthropic.types import Message, MessageDeltaEvent, MessageStartEvent, Usage
from anthropic.types.message_delta_usage import MessageDeltaUsage
from anthropic.types.raw_message_delta_event import Delta

from microcore.llm.anthropic import _process_streamed_response, _update_stream_usage
from microcore.llm.openai import _process_streamed_response as oai_process_streamed_response
from microcore.llm.shared import (
    ensure_stream_include_usage,
    normalize_usage,
    streaming_usage_attrs,
)


def test_normalize_usage():
    oai = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert normalize_usage(oai) == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    anthropic = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 5,
    }
    assert normalize_usage(anthropic) == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 5,
        "cache_included_in_prompt": False,
    }

    google = SimpleNamespace(prompt_token_count=7, candidates_token_count=3)
    assert normalize_usage(google) == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }

    assert normalize_usage(None) is None
    assert normalize_usage({}) is None

    assert normalize_usage({"prompt_tokens": None, "input_tokens": 5}) == {
        "prompt_tokens": 5,
        "completion_tokens": None,
        "total_tokens": 5,
    }
    assert normalize_usage(
        SimpleNamespace(prompt_tokens=None, input_tokens=5)
    ) == {
        "prompt_tokens": 5,
        "completion_tokens": None,
        "total_tokens": 5,
    }


def test_normalize_usage_openai_nested_details():
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {
            "cached_tokens": 300,
            "cache_write_tokens": 100,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 50,
        },
    }
    assert normalize_usage(usage) == {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 100,
        "reasoning_tokens": 50,
        "cache_included_in_prompt": True,
    }


def test_normalize_usage_openai_responses_details():
    usage = {
        "input_tokens": 100,
        "output_tokens": 40,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 10},
    }
    assert normalize_usage(usage) == {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "cache_read_input_tokens": 20,
        "reasoning_tokens": 10,
        "cache_included_in_prompt": True,
    }


def test_normalize_usage_gemini_camel_case():
    usage = {
        "promptTokenCount": 80,
        "candidatesTokenCount": 20,
        "totalTokenCount": 100,
        "cachedContentTokenCount": 15,
        "thoughtsTokenCount": 5,
    }
    assert normalize_usage(usage) == {
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 100,
        "cache_read_input_tokens": 15,
        "reasoning_tokens": 5,
        "cache_included_in_prompt": True,
    }


def test_normalize_usage_gemini_sdk_snake_case():
    # google.genai UsageMetadata uses snake_case attribute names.
    usage = SimpleNamespace(
        prompt_token_count=80,
        candidates_token_count=20,
        total_token_count=100,
        cached_content_token_count=15,
        thoughts_token_count=5,
    )
    assert normalize_usage(usage) == {
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 100,
        "cache_read_input_tokens": 15,
        "reasoning_tokens": 5,
        "cache_included_in_prompt": True,
    }


def test_normalize_usage_idempotent_preserves_cache_flag():
    # After the first pass Anthropic usage has prompt_tokens (not input_tokens),
    # so re-inferring the flag would wrongly flip it to True.
    raw = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 5,
    }
    once = normalize_usage(raw)
    assert once["cache_included_in_prompt"] is False
    twice = normalize_usage(once)
    assert twice == once
    assert twice["cache_included_in_prompt"] is False


@pytest.mark.parametrize(
    "initial,expected",
    [
        ({}, {"include_usage": True}),
        ({"stream_options": {}}, {"include_usage": True}),
        ({"stream_options": {"include_usage": False}}, {"include_usage": False}),
        ({"stream_options": {"foo": 1}}, {"foo": 1, "include_usage": True}),
    ],
)
def test_ensure_stream_include_usage(initial, expected):
    args = {"stream": True, **initial}
    ensure_stream_include_usage(args)
    assert args["stream_options"] == expected


def test_streaming_usage_attrs():
    assert streaming_usage_attrs(None) == {}
    assert streaming_usage_attrs({"input_tokens": 1, "output_tokens": 2}) == {
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }
    }
    assert streaming_usage_attrs(
        {
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_read_input_tokens": 3,
        }
    ) == {
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "cache_read_input_tokens": 3,
            "cache_included_in_prompt": False,
        }
    }


def test_anthropic_streaming_response_usage():
    start_usage = Usage(
        input_tokens=11,
        output_tokens=0,
        cache_read_input_tokens=4,
        cache_creation_input_tokens=2,
    )
    message = Message(
        id="msg_1",
        content=[],
        model="claude-test",
        role="assistant",
        type="message",
        usage=start_usage,
    )
    chunks = [
        MessageStartEvent(type="message_start", message=message),
        MessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="end_turn", stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=9),
        ),
    ]

    usage = {}
    for chunk in chunks:
        usage = _update_stream_usage(chunk, usage)
    assert usage == {
        "input_tokens": 11,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 2,
        "output_tokens": 9,
    }

    response = _process_streamed_response(iter(chunks), [])
    assert response.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 9,
        "total_tokens": 20,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 2,
        "cache_included_in_prompt": False,
    }


def test_openai_streaming_response_usage():
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=1,
                total_tokens=4,
            ),
        ),
    ]
    response = oai_process_streamed_response(iter(chunks), [], chat_model_used=True)
    assert response.content == "Hi"
    assert response.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }
