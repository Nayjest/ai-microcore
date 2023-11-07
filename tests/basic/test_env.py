from microcore import env, configure
from microcore.config import Config


def test_env_default_init():
    assert env().jinja_env is not None
    assert env().config.PROMPT_TEMPLATES_PATH == "tpl"


def test_reinit():
    configure(PROMPT_TEMPLATES_PATH="test1", LLM_API_KEY="123")
    assert env().config.PROMPT_TEMPLATES_PATH == "test1"
    c = Config(PROMPT_TEMPLATES_PATH="test2", LLM_API_KEY="123")
    assert c.PROMPT_TEMPLATES_PATH == "test2"
    configure(**c.__dict__)
    assert env().config.PROMPT_TEMPLATES_PATH == "test2"
    configure()


def test_after_reinit():
    assert env().config.PROMPT_TEMPLATES_PATH == "tpl"
