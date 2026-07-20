from types import SimpleNamespace
from unittest.mock import AsyncMock
import importlib

import pytest

import microcore as mc
from microcore.configuration import Config, LLMApiBaseError, LLMApiKeyError
from microcore.llm.azure_responses import (
    build_responses_client_params,
    is_azure_gpt56_model,
    prepare_responses_args,
    prompt_to_responses_input,
    responses_base_url,
    should_use_azure_responses,
)
from microcore.llm_backends import ApiPlatform

from . import setup  # noqa


def _azure_gpt56_config(**overrides) -> Config:
    params = {
        "LLM_API_TYPE": mc.ApiType.OPENAI,
        "LLM_API_PLATFORM": ApiPlatform.AZURE,
        "LLM_API_KEY": "resource-key",
        "LLM_API_BASE": "https://example.openai.azure.com",
        "LLM_API_VERSION": "2024-06-01",
        "LLM_DEPLOYMENT_ID": "prod-luna",
        "MODEL": "gpt-5.6-luna",
        "VALIDATE_CONFIG": False,
    }
    params.update(overrides)
    return Config(**params)


def _configure_azure_gpt56(**overrides):
    params = {
        "USE_DOT_ENV": False,
        "LLM_API_TYPE": mc.ApiType.OPENAI,
        "LLM_API_PLATFORM": ApiPlatform.AZURE,
        "LLM_API_KEY": "resource-key",
        "LLM_API_BASE": "https://example.openai.azure.com",
        "LLM_API_VERSION": "2024-06-01",
        "LLM_DEPLOYMENT_ID": "prod-luna",
        "MODEL": "gpt-5.6-luna",
        "VALIDATE_CONFIG": False,
    }
    params.update(overrides)
    mc.configure(**params)
    return mc.config()


def test_is_azure_gpt56_model():
    for model in ("gpt-5.6-luna", "GPT-5.6-SOL", "gpt-5.6"):
        assert is_azure_gpt56_model(ApiPlatform.AZURE, model)
        assert is_azure_gpt56_model("azure", model)
    for model in ("gpt-5.5", "gpt-5.60", ""):
        assert not is_azure_gpt56_model(ApiPlatform.AZURE, model)
    assert not is_azure_gpt56_model(ApiPlatform.OPENAI, "gpt-5.6-luna")


def test_should_use_azure_responses_only_for_azure_gpt_56():
    for model in ("gpt-5.6-luna", "GPT-5.6-SOL", "gpt-5.6"):
        cfg = _azure_gpt56_config(MODEL=model)
        assert should_use_azure_responses(cfg)
    for model in ("gpt-5.5", "gpt-5.60", ""):
        cfg = _azure_gpt56_config(MODEL=model)
        assert not should_use_azure_responses(cfg)
    cfg = _azure_gpt56_config(MODEL="gpt-4o")
    assert not should_use_azure_responses(cfg)
    assert not should_use_azure_responses(
        Config(
            LLM_API_PLATFORM=ApiPlatform.OPENAI,
            MODEL="gpt-5.6-luna",
            VALIDATE_CONFIG=False,
        )
    )


def test_responses_base_url_is_normalized():
    assert responses_base_url("https://example.openai.azure.com/") == (
        "https://example.openai.azure.com/openai/v1/"
    )
    assert responses_base_url("https://example.services.ai.azure.com/openai/v1/") == (
        "https://example.services.ai.azure.com/openai/v1/"
    )
    assert responses_base_url("https://example.openai.azure.com/openai") == (
        "https://example.openai.azure.com/openai/v1/"
    )
    assert responses_base_url(
        "https://example.openai.azure.com/openai/deployments/prod-luna"
    ) == (
        "https://example.openai.azure.com/openai/v1/"
    )


def test_responses_base_url_requires_endpoint():
    with pytest.raises(
        LLMApiBaseError,
        match="API Base URL is missing. Please enter a valid API Base URL.",
    ):
        responses_base_url("")


def test_build_responses_client_params_uses_api_key_only():
    cfg = _azure_gpt56_config()
    params = build_responses_client_params(cfg)
    assert params == {
        "api_key": "resource-key",
        "base_url": "https://example.openai.azure.com/openai/v1/",
    }


def test_build_responses_client_params_requires_credentials():
    cfg = _azure_gpt56_config(LLM_API_KEY="")
    with pytest.raises(LLMApiKeyError, match="API Key is missing"):
        build_responses_client_params(cfg)


def test_prepare_responses_args_maps_reasoning_and_defaults():
    args = prepare_responses_args(
        {
            "model": "prod-luna",
            "reasoning_effort": "medium",
            "seed": 42,
            "stream_options": {"include_usage": True},
            "extra_body": {"data_sources": []},
        },
    )
    assert args["model"] == "prod-luna"
    assert args["reasoning"] == {"effort": "medium"}
    assert args["store"] is False
    assert args["include"] == ["reasoning.encrypted_content"]
    assert "seed" not in args
    assert "stream_options" not in args
    assert "extra_body" not in args


def test_prepare_responses_args_default_reasoning_effort_is_medium():
    args = prepare_responses_args({"model": "prod-luna"})
    assert args["reasoning"] == {"effort": "medium"}


def test_prepare_responses_args_respects_llm_default_args():
    args = prepare_responses_args({"model": "prod-luna", "reasoning_effort": "low"})
    assert args["reasoning"] == {"effort": "low"}


def test_prompt_to_responses_input():
    assert prompt_to_responses_input("hello", lambda p: []) == "hello"
    converted = prompt_to_responses_input(
        ["hello"],
        lambda p: [{"role": "user", "content": "hello"}],
    )
    assert converted == "hello"
    converted = prompt_to_responses_input(
        ["system", "user"],
        lambda p: [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    )
    assert converted == [
        {"type": "message", "role": "system", "content": "system"},
        {"type": "message", "role": "user", "content": "user"},
    ]


@pytest.mark.asyncio
async def test_allm_azure_gpt56_uses_responses_api(setup, mocker):
    responses_create = mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(output_text="test successful", usage=None),
    )
    chat_create = mocker.patch(
        "openai.resources.chat.AsyncCompletions.create",
        new_callable=AsyncMock,
    )
    _configure_azure_gpt56()

    result = await mc.allm("test successful")

    assert str(result) == "test successful"
    responses_create.assert_awaited_once()
    chat_create.assert_not_called()
    assert responses_create.await_args.kwargs["model"] == "prod-luna"
    assert responses_create.await_args.kwargs["input"] == "test successful"
    assert responses_create.await_args.kwargs["reasoning"] == {"effort": "medium"}
    assert responses_create.await_args.kwargs["store"] is False
    assert responses_create.await_args.kwargs["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
async def test_allm_azure_gpt55_uses_chat_completions(setup, mocker):
    responses_create = mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
    )
    chat_create = mocker.patch(
        "openai.resources.chat.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
            __dict__={"usage": None},
        ),
    )
    _configure_azure_gpt56(MODEL="gpt-5.5", LLM_DEPLOYMENT_ID="gpt-55")

    result = await mc.allm("ok")

    assert str(result) == "ok"
    chat_create.assert_awaited_once()
    responses_create.assert_not_called()


@pytest.mark.asyncio
async def test_allm_azure_gpt56_streaming_callbacks(setup, mocker):
    async def stream():
        yield SimpleNamespace(type="response.output_text.delta", delta="hel")
        yield SimpleNamespace(type="response.output_text.delta", delta="lo")
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage=None),
        )

    mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
        return_value=stream(),
    )
    _configure_azure_gpt56()

    chunks = []

    async def handler(chunk):
        chunks.append(chunk)

    result = await mc.allm("hello", callback=handler)

    assert str(result) == "hello"
    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_allm_azure_gpt56_multi_turn_input(setup, mocker):
    responses_create = mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(output_text="done", usage=None),
    )
    _configure_azure_gpt56()

    await mc.allm([
        mc.SysMsg("You are helpful."),
        mc.UserMsg("Hello"),
    ])

    assert responses_create.await_args.kwargs["input"] == [
        {"type": "message", "role": "system", "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "Hello"},
    ]


def test_responses_client_uses_openai_base_url(setup, mocker):
    _configure_azure_gpt56()
    openai_module = importlib.import_module("microcore.llm.openai")
    constructor = mocker.patch.object(openai_module.openai, "OpenAI")
    mocker.patch.object(openai_module.openai, "AsyncOpenAI")

    OpenAIClient = openai_module.OpenAIClient
    OpenAIClient(mc.config())

    constructor.assert_called_once_with(
        api_key="resource-key",
        base_url="https://example.openai.azure.com/openai/v1/",
    )
