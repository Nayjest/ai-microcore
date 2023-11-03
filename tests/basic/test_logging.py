from microcore import use_logging, env


def test_logging():
    assert not env().llm_before_handlers
    assert not env().llm_after_handlers
    use_logging()
    assert 1 == len(env().llm_before_handlers)
    assert 1 == len(env().llm_after_handlers)
    use_logging()
    assert 1 == len(env().llm_before_handlers)
    assert 1 == len(env().llm_after_handlers)
