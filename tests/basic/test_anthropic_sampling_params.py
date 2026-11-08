import pytest

from microcore import ApiType, Config
from microcore.llm.anthropic import _accepts_sampling_params, _prepare_llm_arguments


@pytest.mark.parametrize(
    "model, accepts",
    [
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-6", True),
        ("claude-haiku-4-5-20251001", True),
        ("claude-opus-4-20250514", True),
        ("claude-3-7-sonnet-20250219", True),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("some-other-model", True),
        ("claude-opus-4-7", False),
        ("anthropic/claude-opus-4.8", False),
        ("claude-opus-4-7@20260416", False),
        ("claude-sonnet-5", False),
        ("claude-haiku-5-5", False),
        ("claude-fable-5-1", False),
        ("claude-mythos-5", False),
        ("claude-mythos-preview", False),
    ],
)
def test_accepts_sampling_params(model, accepts):
    assert _accepts_sampling_params(model, "2023-06-01") is accepts
    assert _accepts_sampling_params(model, "2026-10-21") is False


@pytest.mark.parametrize(
    "model, api_version, kept",
    [
        ("claude-sonnet-4-6", "2023-06-01", True),
        ("claude-opus-4-7", "2023-06-01", False),
        ("claude-sonnet-4-6", "2026-10-21", False),
    ],
)
def test_prepare_llm_arguments_sampling_params(model, api_version, kept):
    config = Config(
        LLM_API_TYPE=ApiType.ANTHROPIC,
        LLM_API_KEY="-",
        MODEL=model,
        USE_DOT_ENV=False,
        LLM_DEFAULT_ARGS={"temperature": 0.2, "top_k": 5},
    )
    args, _ = _prepare_llm_arguments(config, {}, api_version)
    assert {"temperature" in args, "top_k" in args} == {kept}
