import pytest

from .setup_env import setup_env  # noqa
import microcore


@pytest.mark.asyncio
async def test_streaming_count_async(setup_env):  # noqa
    microcore.use_logging()
    out = []

    def handler(chunk):
        out.append(chunk)

    await microcore.allm(
        "Count from one to ten with english words (like one, two, ...)",
        callbacks=[handler],
    )
    assert "three" in "".join(out).lower()


@pytest.mark.asyncio
async def test_streaming_count_async_async(setup_env):  # noqa
    microcore.use_logging()
    out = []

    async def ahandler(chunk):
        out.append(chunk)

    await microcore.allm(
        "Count from one to ten with english words (like one, two, ...)",
        callbacks=[ahandler],
    )
    assert "six" in "".join(out).lower()
