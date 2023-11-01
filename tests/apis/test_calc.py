from microcore import SysMsg, UserMsg, AssistantMsg, llm

from .setup_env import setup_env # noqa


def test_calc(setup_env): # noqa
    assert '7' == llm([
        SysMsg('You are a calculator'),
        UserMsg('1+2='),
        AssistantMsg('3'),
        UserMsg('3+4=')]
    ).strip()
