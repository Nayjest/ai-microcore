import json

from microcore import tpl, prompt
from . import *  # noqa


def test_tpl(setup):  # noqa
    prompt_str = tpl("test.j2", var="val1")
    assert prompt_str == "Test template val1"
    assert prompt_str.tpl_file == "test.j2"
    assert prompt_str.tpl_vars["var"] == "val1"
    assert tpl("test.j2", var="val2") == "Test template val2"
    assert tpl("test.j2", var2="val2") == "Test template "


def test_tpl_json(setup):  # noqa
    assert json.loads(tpl("json_data.j2", var="val1"))["data"] == "val1"


def test_tpl_subfolder(setup):  # noqa
    prompt_str = tpl("sub-folder/tpl.j2")
    assert prompt_str == "tpl from sub-folder"
    assert prompt_str.tpl_file == "sub-folder/tpl.j2"
    assert prompt_str.tpl_vars == {}



def test_prompt(setup):  # noqa
    prompt_str = prompt("Hi {{ username }}", username="John")
    assert prompt_str == "Hi John"
    assert prompt_str.tpl_vars["username"] == "John"
    assert prompt_str.tpl_file is None
    prompt_str = prompt("Hi")
    assert prompt_str == "Hi"
    assert prompt_str.tpl_vars == {}
    assert prompt_str.tpl_file is None
