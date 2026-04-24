import os
import asyncio
import pytest

import microcore as mc


def test_valid():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(USE_DOT_ENV=False, LLM_API_KEY="123")
    mc.validate_config()
    assert not mc.env().config.uses_local_model()
    os.environ.update(original_env)


def test_no_key():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(USE_DOT_ENV=False)
    os.environ.update(original_env)


def test_azure_no_deployment():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(
            USE_DOT_ENV=False,
            LLM_API_TYPE=mc.ApiType.OPENAI,
            LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
            LLM_API_KEY="123",
            LLM_API_VERSION="123",
            LLM_API_BASE="https://example.com",
        )
        mc.validate_config()
    os.environ.update(original_env)


def test_azure_ok():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
        LLM_API_KEY="123",
        LLM_DEPLOYMENT_ID="123",
        LLM_API_VERSION="123",
        LLM_API_BASE="https://example.com",
    )

    mc.validate_config()
    assert not mc.env().config.uses_local_model()
    os.environ.update(original_env)


def test_azure_entra_ok_without_api_key():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    try:
        mc.configure(
            USE_DOT_ENV=False,
            LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
            LLM_AZURE_USE_ENTRA_ID=True,
            LLM_DEPLOYMENT_ID="dep",
            LLM_API_VERSION="2024-02-15-preview",
            LLM_API_BASE="https://example.openai.azure.com",
        )
        mc.validate_config()
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_azure_entra_invalid_credential_mode():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    try:
        with pytest.raises(mc.LLMConfigError):
            mc.configure(
                USE_DOT_ENV=False,
                LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
                LLM_AZURE_USE_ENTRA_ID=True,
                LLM_AZURE_ENTRA_CREDENTIAL="wrong",
                LLM_DEPLOYMENT_ID="dep",
                LLM_API_VERSION="2024-02-15-preview",
                LLM_API_BASE="https://example.openai.azure.com",
            )
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_azure_entra_ok_from_process_env():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    try:
        os.environ["LLM_API_TYPE"] = mc.ApiType.OPENAI
        os.environ["LLM_API_PLATFORM"] = mc.ApiPlatform.AZURE
        os.environ["LLM_API_BASE"] = "https://example.openai.azure.com"
        os.environ["LLM_DEPLOYMENT_ID"] = "dep"
        os.environ["LLM_API_VERSION"] = "2024-02-15-preview"
        os.environ["MODEL"] = "dep"
        os.environ["LLM_AZURE_USE_ENTRA_ID"] = "true"
        mc.configure(USE_DOT_ENV=False)
        mc.validate_config()
        assert mc.env().config.LLM_AZURE_ENTRA_SCOPE == "https://ai.azure.com/.default"
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_azure_no_version():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(
            LLM_DEPLOYMENT_ID="123",
            USE_DOT_ENV=False,
            LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
            LLM_API_KEY="123",
            LLM_API_BASE="https://example.com",
        )
        mc.validate_config()

    os.environ.update(original_env)


def test_local_llm():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()

    def inference(prompt, **kwargs):
        return prompt

    mc.configure(
        USE_DOT_ENV=False, LLM_API_TYPE=mc.ApiType.FUNCTION, INFERENCE_FUNC=inference
    )
    assert mc.env().config.uses_local_model()
    assert mc.llm("test") == "test"
    mc.configure(USE_DOT_ENV=False, INFERENCE_FUNC=lambda x: x + ":1")
    assert mc.llm("test") == "test:1"
    assert mc.env().config.uses_local_model()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(
            USE_DOT_ENV=False,
            LLM_API_TYPE=mc.ApiType.FUNCTION,
        )

    os.environ.update(original_env)


def test_none():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.NONE,
    )
    assert mc.env().config.uses_local_model()
    with pytest.raises(mc.LLMConfigError):
        mc.llm("test")
    with pytest.raises(mc.LLMConfigError):

        async def fn():
            await asyncio.sleep(0.0001)
            await mc.allm("test")

        asyncio.run(fn())
    os.environ.update(original_env)
