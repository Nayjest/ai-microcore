from microcore.utils import resolve_callable, CantResolveCallable
from microcore.ui import red
import pytest

def test_resolve_callable():
    assert resolve_callable("microcore.ui.red")() == red()
    assert resolve_callable("tests.basic.fixtures.CapsFixture.fn")() == "fn_result"
    assert resolve_callable(
        "tests.basic.fixtures.CapsFixture.TestClass.fn"
    )() == "TestClass.fn_result"

    with pytest.raises(CantResolveCallable):
        resolve_callable("non.existent.module.function")

    with pytest.raises(CantResolveCallable):
        resolve_callable("microcore.ui.non_ex_function")
