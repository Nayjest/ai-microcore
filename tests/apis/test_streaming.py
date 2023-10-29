from .setup_env import setup_env # noqa
import microcore


def test_streaming_count(setup_env): # noqa
    microcore.use_logging()
    out = []

    def handler(chunk):
        out.append(chunk)
    microcore.llm(
        'Count from one to twenty with english words (like one, two, ...)',
        callback=handler
    )
    assert 'three' ''.join(out).lower()
