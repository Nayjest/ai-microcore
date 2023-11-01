
from . import * # noqa


def test_tpl(setup):
    mc.configure(MODEL='gpt-4')
    assert mc.llm('ok') == 'ok'
    mc.configure(MODEL='gpt-3.5-instruct')
    assert mc.llm('ok') == 'completion:ok'
