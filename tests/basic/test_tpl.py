import json

from microcore import tpl
from . import *  # noqa


def test_tpl(setup):  # noqa
    assert tpl("test.j2", var="val1") == "Test template val1"
    assert tpl("test.j2", var="val2") == "Test template val2"
    assert tpl("test.j2", var2="val2") == "Test template "


def test_tpl_json(setup):  # noqa
    assert json.loads(tpl("json_data.j2", var="val1"))["data"] == "val1"


def test_tpl_subfolder(setup):  # noqa
    assert tpl("sub-folder/tpl.j2") == "tpl from sub-folder"
