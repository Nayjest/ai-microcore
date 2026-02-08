from microcore import SysMsg, UserMsg, AssistantMsg, llm, use_logging

from .setup_env import setup_env  # noqa


def test_calc(setup_env):  # noqa
    use_logging()
    assert (
        7 == llm(
            [
                SysMsg("You are a calculator, answer with number"),
                UserMsg("1+2="),
                AssistantMsg("3"),
                UserMsg("3+4="),
            ]
        ).parse_number(dtype=int, position="last")
    )
