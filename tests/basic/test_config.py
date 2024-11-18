import pytest

import microcore as mc
from microcore.configuration import Config
from dataclasses import asdict


def test_config(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "123")
    monkeypatch.setenv("PROMPT_TEMPLATES_PATH", "mypath")
    assert Config().PROMPT_TEMPLATES_PATH == "mypath"
    monkeypatch.delenv("PROMPT_TEMPLATES_PATH", raising=False)
    assert Config().PROMPT_TEMPLATES_PATH == "tpl"


def test_config_from_inst(monkeypatch):
    config1 = Config(LLM_API_KEY="KEY1")
    config2 = Config(LLM_API_KEY="KEY2", MODEL="MODEL2")
    config3 = Config(LLM_API_KEY="KEY3")
    mc.configure(**config1.__dict__)
    assert mc.config().LLM_API_KEY == "KEY1"
    mc.configure(**dict(config2))
    assert mc.config().LLM_API_KEY == "KEY2"
    mc.configure(**asdict(config3))
    assert mc.config().LLM_API_KEY == "KEY3"
    assert mc.config().MODEL != "MODEL2"
    mc.configure(LLM_API_KEY="KEY4")
    assert mc.config().LLM_API_KEY == "KEY4"


def test_config_wrong_key(monkeypatch):
    with pytest.raises(TypeError):
        Config(WRONG_KEY="KEY1", LLM_API_KEY="KEY2")


def test_config_case_convert(monkeypatch):
    assert mc.configure(llm_api_key="k1").LLM_API_KEY == "k1"


def test_config_key_prefixing(monkeypatch):
    assert mc.configure(api_key="k2").LLM_API_KEY == "k2"


def test_config_from_dict(monkeypatch):
    assert mc.configure(dict(api_key="k3")).LLM_API_KEY == "k3"


def test_config_from_config(monkeypatch):
    assert mc.configure(Config(LLM_API_KEY="k4")).LLM_API_KEY == "k4"


def test_dataclass_fields_prefixing(monkeypatch):
    assert (
        mc.configure(
            api_type=mc.ApiType.NONE,
            default_args={"max_new_tokens": 77},
        ).LLM_DEFAULT_ARGS["max_new_tokens"]
        == 77
    )
