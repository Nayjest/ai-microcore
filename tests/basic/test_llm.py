import pytest

from . import setup  # noqa
import microcore as mc


@pytest.mark.asyncio
async def test_llm_mocked_parrot(setup):
    assert mc.llm("ok", model="gpt-4") == "ok"
    assert mc.llm("ok", model="gpt-3.5-instruct") == "completion:ok"
    mc.configure(USE_DOT_ENV=False, LLM_API_KEY="123", MODEL="some-chat-llama")
    assert mc.llm("ok") == "ok"
    assert mc.llm("ok", model="text-davinci-003") == "completion:ok"
    mc.configure(USE_DOT_ENV=False, LLM_API_KEY="123", MODEL="text-curie-001")
    assert mc.llm("ok") == "completion:ok"
    assert mc.llm("ok", model="chat-llama") == "ok"
    assert await mc.allm("ok", model="gpt-4") == "ok"
    assert await mc.allm("ok", model="gpt-3.5-instruct") == "completion:ok"


@pytest.mark.asyncio
async def test_llm_no_streaming(setup):
    t = ""

    def fn(text):
        nonlocal t
        t += text

    mc.llm("ok", model="gpt-4", stream=False, callback=fn)
    assert t == "ok"

    def afn(text):
        nonlocal t
        t += text

    t = ""
    mc.llm("ok2", model="gpt-4", stream=False, callback=afn)
    assert t == "ok2"

    t = ""
    mc.llm("ok", model="gpt-4", stream=False, callbacks=[afn, fn])
    assert t == "okok"



def test_llm_with_parsing_json_and_retries(setup):
    t = ""

    def fn(text):
        nonlocal t
        t += text

    def afn(text):
        nonlocal t
        t += text

    with pytest.raises(mc.BadAIJsonAnswer):
        mc.llm('{a=1-2}', retries=2, stream=False, callbacks=[fn,afn]).parse_json()
    assert t == '{a=1-2}'*3*2

    t = ""
    with pytest.raises(mc.BadAIJsonAnswer):
        mc.llm('{a=1-2}', retries=3, parse_json=True, callback=afn, stream=False)
    assert t == '{a=1-2}' * 4

    assert mc.llm('{a=1-2}').parse_json(raise_errors=False) == False

    assert mc.llm('{a=1-2}', retries=2).parse_json(raise_errors=False) == False