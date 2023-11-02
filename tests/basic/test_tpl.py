from microcore import tpl
from . import *  # noqa


def test_tpl(setup):  # noqa
    assert tpl("test.j2", var="val1") == "Test template val1"
    assert tpl("test.j2", var="val2") == "Test template val2"
