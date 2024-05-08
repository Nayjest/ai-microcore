import time

import pytest

from microcore.metrics import Metrics
from . import setup  # noqa
import microcore as mc


@pytest.mark.asyncio
async def test_metrics(setup):
    with Metrics() as m:
        assert mc.llm("ok", model="gpt-4") == "ok"
        assert mc.llm("ok", model="gpt-3.5-instruct") == "completion:ok"
        assert await mc.allm("ok", model="gpt-4") == "ok"
        assert await mc.allm("ok", model="gpt-3.5-instruct") == "completion:ok"
        time.sleep(0.5)
    assert m.requests_count == 4
    assert m.succ_requests_count == 4
    assert m.gen_chars_count == len(("ok"+"completion:ok")*2), "Incorrect gen_chars_count"
    assert 0.5 < m.exec_duration < 1
