from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import asyncio
import importlib

import pytest

import microcore as mc
from microcore.configuration import Config, LLMApiBaseError, LLMApiKeyError
from microcore.llm.azure_responses import (
    build_responses_client_params,
    prepare_responses_args,
    prompt_to_responses_input,
    responses_base_url,
    should_use_responses_api,
)
from microcore.llm_backends import ApiPlatform

from . import setup  # noqa


_AZURE_GPT56 = {
    "LLM_API_TYPE": mc.ApiType.OPENAI,
    "LLM_API_PLATFORM": ApiPlatform.AZURE,
    "LLM_API_KEY": "resource-key",
    "LLM_API_BASE": "https://example.openai.azure.com",
    "LLM_API_VERSION": "2024-06-01",
    "LLM_DEPLOYMENT_ID": "prod-luna",
    "MODEL": "gpt-5.6-luna",
    "VALIDATE_CONFIG": False,
}

_OPENAI = {
    "LLM_API_PLATFORM": ApiPlatform.OPENAI,
    "LLM_API_KEY": "sk-test",
    "MODEL": "gpt-4o",
    "LLM_API_BASE": "https://api.openai.com/v1",
    "LLM_DEPLOYMENT_ID": "",
    "LLM_API_VERSION": "",
}


def _config(**overrides):
    """Build a Config without constructing an OpenAI client."""
    return Config(**{**_AZURE_GPT56, **overrides})


def _configure(**overrides):
    mc.configure(USE_DOT_ENV=False, **{**_AZURE_GPT56, **overrides})
    return mc.config()


def _chat_completion(content="chat-ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
        __dict__={"usage": None},
    )


def test_should_use_responses_api_config_and_override():
    # default (unset) -> Chat Completions, no model-name magic
    assert should_use_responses_api(_config()) is False
    assert should_use_responses_api(_config(LLM_USE_RESPONSES_API=False)) is False
    # config flag opts in
    assert should_use_responses_api(_config(LLM_USE_RESPONSES_API=True)) is True
    # per-request override wins over config
    assert should_use_responses_api(
        _config(LLM_USE_RESPONSES_API=False), override=True
    ) is True
    assert should_use_responses_api(
        _config(LLM_USE_RESPONSES_API=True), override=False
    ) is False


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
    cfg = _config()
    params = build_responses_client_params(cfg)
    assert params == {
        "api_key": "resource-key",
        "base_url": "https://example.openai.azure.com/openai/v1/",
    }


def test_build_responses_client_params_requires_credentials():
    cfg = _config(LLM_API_KEY="")
    with pytest.raises(LLMApiKeyError, match="API Key is missing"):
        build_responses_client_params(cfg)


def test_build_responses_client_params_uses_entra_token_provider_callable():
    cfg = _config(LLM_API_KEY="")
    token_provider = lambda: "fresh-token"  # noqa: E731

    params = build_responses_client_params(cfg, entra_token_provider=token_provider)

    assert params["api_key"] is token_provider
    assert params["base_url"] == "https://example.openai.azure.com/openai/v1/"
    assert "default_headers" not in params


def test_build_responses_client_params_uses_api_key_from_headers():
    cfg = _config(
        LLM_API_KEY="",
        HTTP_HEADERS={"api-key": "header-key"},
    )
    params = build_responses_client_params(cfg)
    assert params["api_key"] == "header-key"
    assert params["default_headers"] == {"api-key": "header-key"}


def test_build_responses_client_params_uses_api_key_from_init_params():
    cfg = _config(
        LLM_API_KEY="",
        INIT_PARAMS={"api_key": "init-key"},
    )
    params = build_responses_client_params(cfg)
    assert params["api_key"] == "init-key"


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


def test_prepare_responses_args_respects_explicit_reasoning_effort():
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
    _configure(LLM_USE_RESPONSES_API=True)

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
async def test_allm_azure_gpt56_raises_on_responses_error(setup, mocker):
    mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(error="quota exceeded", output_text=""),
    )
    _configure(LLM_USE_RESPONSES_API=True)

    with pytest.raises(mc.BadAIAnswer, match="quota exceeded"):
        await mc.allm("hello")


@pytest.mark.asyncio
async def test_allm_openai_config_flag_enables_responses(setup, mocker):
    responses_create = mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(output_text="resp-ok", usage=None),
    )
    chat_create = mocker.patch(
        "openai.resources.chat.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=_chat_completion(),
    )
    _configure(**_OPENAI, LLM_USE_RESPONSES_API=True)

    result = await mc.allm("hi")

    assert str(result) == "resp-ok"
    responses_create.assert_awaited_once()
    chat_create.assert_not_called()


@pytest.mark.asyncio
async def test_allm_per_request_use_responses_api_true(setup, mocker):
    responses_create = mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(output_text="resp-ok", usage=None),
    )
    _configure(**_OPENAI)  # auto -> Chat Completions

    result = await mc.allm("hi", use_responses_api=True)

    assert str(result) == "resp-ok"
    responses_create.assert_awaited_once()
    # request-level flag must not leak into the API payload
    assert "use_responses_api" not in responses_create.await_args.kwargs


@pytest.mark.asyncio
async def test_allm_azure_gpt56_per_request_opt_out_uses_chat(setup, mocker):
    responses_create = mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
    )
    chat_create = mocker.patch(
        "openai.resources.chat.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=_chat_completion(),
    )
    _configure(LLM_USE_RESPONSES_API=True)

    # Same process, same v1 client, but this call opts out to Chat Completions
    result = await mc.allm("hi", use_responses_api=False)

    assert str(result) == "chat-ok"
    chat_create.assert_awaited_once()
    responses_create.assert_not_called()
    assert "use_responses_api" not in chat_create.await_args.kwargs


@pytest.mark.asyncio
async def test_allm_config_false_disables_responses_on_azure_gpt56(setup, mocker):
    responses_create = mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
    )
    chat_create = mocker.patch(
        "openai.resources.chat.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=_chat_completion(),
    )
    _configure(LLM_USE_RESPONSES_API=False)

    result = await mc.allm("hi")

    assert str(result) == "chat-ok"
    responses_create.assert_not_called()


@pytest.mark.asyncio
async def test_allm_per_request_responses_unavailable_raises(setup, mocker):
    mocker.patch(
        "openai.resources.responses.AsyncResponses.create",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "openai.resources.chat.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=_chat_completion(),
    )
    # classic AzureOpenAI client (no v1 endpoint) cannot serve Responses
    _configure(LLM_USE_RESPONSES_API=False)

    with pytest.raises(mc.LLMConfigError):
        await mc.allm("hi", use_responses_api=True)


@pytest.mark.asyncio
async def test_allm_default_uses_chat_completions(setup, mocker):
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
    _configure()  # no flag -> Chat Completions regardless of model

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
    _configure(LLM_USE_RESPONSES_API=True)

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
    _configure(LLM_USE_RESPONSES_API=True)

    await mc.allm([
        mc.SysMsg("You are helpful."),
        mc.UserMsg("Hello"),
    ])

    assert responses_create.await_args.kwargs["input"] == [
        {"type": "message", "role": "system", "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "Hello"},
    ]


def test_responses_client_uses_openai_base_url(setup, mocker):
    _configure(LLM_USE_RESPONSES_API=True)
    openai_module = importlib.import_module("microcore.llm.openai")
    constructor = mocker.patch.object(openai_module.openai, "OpenAI")
    mocker.patch.object(openai_module.openai, "AsyncOpenAI")

    OpenAIClient = openai_module.OpenAIClient
    OpenAIClient(mc.config())

    constructor.assert_called_once_with(
        api_key="resource-key",
        base_url="https://example.openai.azure.com/openai/v1/",
    )


def test_async_client_wraps_sync_entra_api_key_provider():
    """openai 1.109+ awaits api_key; sync Entra provider must be wrapped."""
    openai_module = importlib.import_module("microcore.llm.openai")
    cfg = _config(LLM_USE_RESPONSES_API=True, LLM_AZURE_USE_ENTRA_ID=True)
    with (
        patch.object(openai_module.openai, "OpenAI"),
        patch.object(openai_module.openai, "AsyncOpenAI") as async_ctor,
        patch.object(
            openai_module,
            "_build_azure_entra_token_provider",
            return_value=lambda: "entra-token",
        ),
    ):
        openai_module.OpenAIClient(cfg)
        async_key = async_ctor.call_args.kwargs["api_key"]
        assert asyncio.iscoroutinefunction(async_key)
        assert asyncio.run(async_key()) == "entra-token"


def test_wrapped_api_key_survives_openai_refresh_api_key():
    """Regression for openai 1.109: await _api_key_provider() must not see a str."""
    import openai
    from microcore.llm.openai import _as_async_str_provider

    async def _run():
        client = openai.AsyncOpenAI(
            api_key=_as_async_str_provider(lambda: "entra-token"),
            base_url="https://example.invalid/v1",
        )
        await client._refresh_api_key()
        assert client.api_key == "entra-token"

    asyncio.run(_run())
