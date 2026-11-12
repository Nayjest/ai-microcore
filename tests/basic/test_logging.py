import pytest

import microcore as mc
from microcore import use_logging, env, config
from microcore.logging import LoggingConfig, _stream_log_end


def test_logging():
    assert not env().llm_before_handlers
    assert not env().llm_after_handlers
    use_logging()
    assert 1 == len(env().llm_before_handlers)
    assert 1 == len(env().llm_after_handlers)
    use_logging()
    assert 1 == len(env().llm_before_handlers)
    assert 1 == len(env().llm_after_handlers)


def test_stream_logging():
    use_logging(stream=True)
    assert 1 == len(env().llm_before_handlers)
    assert 1 == len(config().CALLBACKS)
    # closes the line of the streamed response
    assert [_stream_log_end] == env().llm_after_handlers
    use_logging(stream=True)
    assert 1 == len(env().llm_before_handlers)
    assert 1 == len(config().CALLBACKS)
    assert [_stream_log_end] == env().llm_after_handlers
    # switching back to non-streaming logging cleans it up
    use_logging()
    assert not config().CALLBACKS
    assert _stream_log_end not in env().llm_after_handlers
    assert 1 == len(env().llm_after_handlers)


def _local_reply(_prompt, **_kwargs):
    return "the cached reply"


@pytest.mark.parametrize("call", [mc.llm, mc.allm])
async def test_print_stream_logs_a_file_cache_hit_once(call, tmp_path, monkeypatch):
    previous = config()
    output = []
    monkeypatch.setattr(mc.storage, "custom_path", str(tmp_path))
    monkeypatch.setattr(LoggingConfig, "OUTPUT_METHOD", output.append)
    try:
        mc.configure(
            USE_DOT_ENV=False,
            LLM_API_TYPE=mc.ApiType.FUNCTION,
            INFERENCE_FUNC=_local_reply,
            MODEL="local-test",
            CHAT_MODE=False,
            USE_LOGGING=mc.PRINT_STREAM,
        )
        LoggingConfig.OUTPUT_METHOD = output.append

        async def ask():
            result = call("question", file_cache=True)
            return await result if call is mc.allm else result

        first = await ask()
        assert first == "the cached reply"
        assert not first.from_file_cache
        assert "".join(output).count("the cached reply") == 1
        output.clear()

        second = await ask()
        assert second == first
        assert second.from_file_cache
        assert "".join(output).count("the cached reply") == 1
        assert output[-1] == "\n"
    finally:
        mc.configure(previous)
