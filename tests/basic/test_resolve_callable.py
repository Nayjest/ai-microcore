from microcore.utils import resolve_callable
from microcore.ui import red


def test_resolve_callable():
    assert resolve_callable("microcore.ui.red")() == red()
    assert resolve_callable("tests.basic.fixtures.CapsFixture.fn")() == "fn_result"
    assert resolve_callable(
        "tests.basic.fixtures.CapsFixture.TestClass.fn"
    )() == "TestClass.fn_result"
