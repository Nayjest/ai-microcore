import microcore as mc
from microcore.message_types import Role
from microcore.wrappers.prompt_wrapper import PromptWrapper

from . import setup


async def test_wrappers(setup):
    out = mc.tpl(file="json_data.j2", var="test_data").to_llm().parse_json()
    assert out == dict(data="test_data")

    out = (await mc.tpl(file="json_data.j2", var="test_data").to_allm()).parse_json()
    assert out == dict(data="test_data")

def test_creation_basic():
    p = PromptWrapper("Hello world")
    assert p == "Hello world"
    assert p.tpl_file is None
    assert p.tpl_vars is None

def test_creation_with_attrs():
    p = PromptWrapper("test", attrs={"key": "value"})
    assert p == "test"
    assert p.key == "value"

def test_creation_with_tpl_file():
    p = PromptWrapper("test", tpl_file="template.txt", tpl_vars={"a": 1})
    assert p.tpl_file == "template.txt"
    assert p.tpl_vars == {"a": 1}

def test_as_message():
    p = PromptWrapper("Hello world")
    assert isinstance(p.as_user, mc.UserMsg)
    assert isinstance(p.as_system, mc.SysMsg)
    assert isinstance(p.as_assistant, mc.AssistantMsg)
    assert p.as_user.content == "Hello world"
    assert p.as_user.role == Role.USER

def test_is_instance_of_str():
    p = PromptWrapper("hello")
    assert isinstance(p, str)
