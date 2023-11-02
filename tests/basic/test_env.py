from microcore import env, configure
from microcore.config import Config


def test_env_default_init(monkeypatch):
    assert env().jinjaEnvironment is not None
    assert env().config.PROMPT_TEMPLATES_PATH == "tpl"


def test_reinit(monkeypatch):
    configure(PROMPT_TEMPLATES_PATH="test1")
    assert env().config.PROMPT_TEMPLATES_PATH == "test1"
    c = Config(PROMPT_TEMPLATES_PATH="test2")
    assert c.PROMPT_TEMPLATES_PATH == "test2"
    configure(**c.__dict__)
    assert env().config.PROMPT_TEMPLATES_PATH == "test2"
    configure()


def test_after_reinit(monkeypatch):
    assert env().config.PROMPT_TEMPLATES_PATH == "tpl"
