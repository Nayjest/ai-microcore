import pytest
from microcore import LLMConfigError, llm, allm, configure, ApiType


def llm_func_parrot(prompt, max_tokens=1024, **kwargs):
    return prompt[:max_tokens]


def test_configure_non_existing():
    with pytest.raises(LLMConfigError):
        configure("tests/basic/config/non-existing")


@pytest.mark.asyncio
async def test_configure_from_file():
    c = configure("tests/basic/config/custom_func.ini")
    assert c.LLM_API_TYPE == ApiType.FUNCTION
    assert c.INIT_PARAMS["quantize_4bit"] is True
    assert c.LLM_DEFAULT_ARGS["max_tokens"] == 5
    assert (
        c.INFERENCE_FUNC
        == "tests.basic.test_configure_from_file_local_func.llm_func_parrot"
    )
    assert llm("ok") == "ok"
    assert llm("123456") == "12345"
    assert await allm("ok") == "ok"
    assert await allm("123456") == "12345"
