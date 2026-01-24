import pytest
import microcore as mc
from .setup_env import setup_env  # noqa

@pytest.mark.asyncio
async def test_stream_parameter(setup_env):  # noqa
    prompt = "List 5 planets from closest to farthest from the Sun (comma-separated, no comments)."
    out = mc.llm(prompt, stream=True).lower()
    assert "mars" in out or "mercury" in out
    out = (await mc.allm(prompt, stream=True)).lower()
    assert "mars" in out or "mercury" in out
    mc.use_logging(stream=True)
    out = (await mc.allm(prompt, stream=True)).lower()
    assert "mars" in out or "mercury" in out
    out = mc.llm(prompt, stream=True).lower()
    assert "mars" in out or "mercury" in out
