from unittest.mock import AsyncMock, Mock

import pytest
from google.genai import models, types

from microcore import ApiPlatform, ApiType, Config
from microcore.llm.google_genai import GoogleClient


@pytest.mark.asyncio
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("max_output_tokens", [None, 4096])
@pytest.mark.parametrize(
    "call_args, expected_level",
    [
        ({}, types.ThinkingLevel.LOW),
        ({"thinking_config": {"thinking_level": "high"}}, types.ThinkingLevel.HIGH),
    ],
)
async def test_google_generation_default_args(
    monkeypatch, async_call, max_output_tokens, call_args, expected_level
):
    defaults = {
        "thinking_config": {"thinking_level": "low"},
        "max_tokens": 512,
        "http_options": {"headers": {"X-Default": "default"}},
    }
    if max_output_tokens is not None:
        defaults["max_output_tokens"] = max_output_tokens
    call_args = {**call_args, "extra_headers": {"X-Request": "request"}}
    client = GoogleClient(
        Config(
            USE_DOT_ENV=False,
            INFERENCE_FUNC=None,
            INIT_PARAMS={},
            LLM_API_TYPE=ApiType.GOOGLE,
            LLM_API_PLATFORM=ApiPlatform.GOOGLE_AI_STUDIO,
            LLM_API_KEY="test-key",
            MODEL="gemini-3.8-flash",
            LLM_DEFAULT_ARGS=defaults,
        )
    )
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(parts=[types.Part(text="{}")]))
        ]
    )
    if async_call:
        generate = AsyncMock(return_value=response)
        monkeypatch.setattr(models.AsyncModels, "generate_content", generate)
        await client.aio.generate("Return JSON", **call_args)
    else:
        generate = Mock(return_value=response)
        monkeypatch.setattr(models.Models, "generate_content", generate)
        client.generate("Return JSON", **call_args)

    generation_config = generate.call_args.kwargs["config"]
    assert generation_config.thinking_config is not None
    assert generation_config.thinking_config.thinking_level == expected_level
    assert generation_config.max_output_tokens == max_output_tokens
    assert generation_config.http_options.headers == {
        "X-Default": "default",
        "X-Request": "request",
    }
    assert client.config.LLM_DEFAULT_ARGS["http_options"]["headers"] == {
        "X-Default": "default",
    }
