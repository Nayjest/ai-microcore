import os
from . import *  # noqa
import asyncio


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
            LLM_API_TYPE=mc.ApiType.AZURE,
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
        LLM_API_TYPE=mc.ApiType.AZURE,
        LLM_API_KEY="123",
        LLM_DEPLOYMENT_ID="123",
        LLM_API_VERSION="123",
        LLM_API_BASE="https://example.com",
    )

    mc.validate_config()
    assert not mc.env().config.uses_local_model()
    os.environ.update(original_env)


def test_azure_no_version():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(
            LLM_DEPLOYMENT_ID="123",
            USE_DOT_ENV=False,
            LLM_API_TYPE=mc.ApiType.AZURE,
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
