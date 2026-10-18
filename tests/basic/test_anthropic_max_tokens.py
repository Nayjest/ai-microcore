import pytest

from microcore import ApiType, Config
from microcore.llm.anthropic import _default_max_tokens, _prepare_llm_arguments


@pytest.mark.parametrize(
    "model, streaming, non_streaming",
    [
        ("claude-opus-5-5", 64_000, 20_000),
        ("claude-sonnet-4-5-20250929", 64_000, 20_000),
        ("claude-opus-4-5-20251101", 64_000, 20_000),
        ("anthropic/claude-opus-4.5", 64_000, 20_000),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", 64_000, 20_000),
        ("claude-sonnet-4-20250514", 64_000, 20_000),
        ("claude-3-7-sonnet-20250219", 64_000, 20_000),
        ("claude-opus-4-20250514", 32_000, 8192),
        ("claude-opus-4-1", 32_000, 8192),
        ("claude-opus-4-1@20250805", 32_000, 8192),
        ("anthropic.claude-opus-4-20250514-v1:0", 32_000, 8192),
        ("anthropic/claude-opus-4.1:beta", 32_000, 8192),
        ("claude-4-opus-20250514", 32_000, 8192),
        ("claude-3-5-sonnet-20241022", 8192, 8192),
        ("anthropic.claude-3-5-haiku-20241022-v1:0", 8192, 8192),
        ("anthropic/claude-3.5-sonnet", 8192, 8192),
        ("claude-3-opus-20240229", 4096, 4096),
        ("claude-3-haiku@20240307", 4096, 4096),
        ("claude-2.1", 4096, 4096),
        ("anthropic.claude-v2:1", 4096, 4096),
        ("claude-instant-1.2", 4096, 4096),
        ("some-other-model", 4096, 4096),
    ],
)
def test_default_max_tokens(model, streaming, non_streaming):
    assert _default_max_tokens(model, stream=True) == streaming
    assert _default_max_tokens(model, stream=False) == non_streaming


def _config(**params):
    return Config(
        LLM_API_TYPE=ApiType.ANTHROPIC,
        LLM_API_KEY="-",
        MODEL="claude-opus-5-5",
        USE_DOT_ENV=False,
        **params,
    )


def test_prepare_llm_arguments_default_max_tokens():
    args, _ = _prepare_llm_arguments(_config(), {})
    assert (args["stream"], args["max_tokens"]) == (False, 20_000)
    # A callback turns streaming on, which allows the larger default
    args, _ = _prepare_llm_arguments(_config(), {"callback": print})
    assert (args["stream"], args["max_tokens"]) == (True, 64_000)


def test_prepare_llm_arguments_explicit_max_tokens():
    config = _config(LLM_DEFAULT_ARGS={"max_tokens": 1000})
    assert _prepare_llm_arguments(config, {})[0]["max_tokens"] == 1000
    assert _prepare_llm_arguments(config, {"max_tokens": 500})[0]["max_tokens"] == 500
