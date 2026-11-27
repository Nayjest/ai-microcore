from microcore import ApiPlatform, ApiType
from microcore.configuration import Config
from microcore.llm_backends import llm_api_base_required, platform_by_api_base


def test_in():  # test str in SafeStrEnum for python < 3.12
    assert "openai" in ApiType
    assert "straitly" in ApiPlatform


def test_straitly_platform(monkeypatch):
    for var in (
        "LLM_API_TYPE",
        "LLM_API_BASE",
        "LLM_API_PLATFORM",
        "INFERENCE_FUNC",
        "LLM_CLI",
        "USE_DOT_ENV",
    ):
        monkeypatch.delenv(var, raising=False)
    assert ApiPlatform.STRAITLY.api_type() == ApiType.OPENAI
    assert ApiPlatform.STRAITLY.default_api_base() == "https://api.straitly.ai/v1"
    assert ApiPlatform.label_for(ApiPlatform.STRAITLY) == "Straitly"
    assert not llm_api_base_required(ApiType.OPENAI, ApiPlatform.STRAITLY)
    assert platform_by_api_base("https://api.straitly.ai/v1") == (
        ApiType.OPENAI,
        ApiPlatform.STRAITLY,
    )

    cfg = Config(
        USE_DOT_ENV=False,
        LLM_API_PLATFORM=ApiPlatform.STRAITLY,
        LLM_API_KEY="test-key",
    )
    assert cfg.LLM_API_TYPE == ApiType.OPENAI
    assert cfg.LLM_API_BASE == "https://api.straitly.ai/v1"
