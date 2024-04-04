import os
import pytest

from microcore import env, configure
from microcore.configuration import Config, LLMConfigError


def test_env_default_init():
    os.environ["LLM_API_KEY"] = "123"
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


def test_failed_reinit():
    os.environ['MODEL'] = 'default-model'
    configure(LLM_API_KEY="old_key", MODEL="old_model")
    # Ensure we configured successfully
    assert env().config.MODEL == "old_model"

    _osenv, os.environ = os.environ, {}
    # Ensure exception is raised on misconfiguration
    with pytest.raises(LLMConfigError):
        configure(LLM_API_KEY="", MODEL="new_model", USE_DOT_ENV=False)
    os.environ = _osenv

    # Ensure we have default values but not the new ones or the old ones
    # after re-configuring with incorrect settings
    assert env().config.MODEL == 'default-model'
