import microcore
from .setup_env import setup_env  # noqa


def test_streaming_count(setup_env):  # noqa
    microcore.use_logging()
    out = []

    def handler(chunk):
        out.append(chunk)

    microcore.llm(
        "Count from one to twenty with english words (like one, two, ...)",
        callback=handler,
    )
    assert "three" in "".join(out).lower()
