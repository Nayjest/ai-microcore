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
