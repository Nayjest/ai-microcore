from microcore import use_logging, env, config
from microcore.logging import _stream_log_end


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
