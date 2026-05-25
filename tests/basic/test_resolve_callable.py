import pytest
from microcore.utils import resolve_callable, CantResolveCallable
from microcore.ui import red


def _global_func():
    return "global_func_result"


def test_resolve_callable():
    assert resolve_callable("microcore.ui.red")() == red()
    assert resolve_callable("tests.basic.fixtures.CapsFixture.fn")() == "fn_result"
    assert (
        resolve_callable("tests.basic.fixtures.CapsFixture.TestClass.fn")()
        == "TestClass.fn_result"
    )

    with pytest.raises(CantResolveCallable):
        resolve_callable("non.existent.module.function")

    with pytest.raises(CantResolveCallable):
        resolve_callable("microcore.ui.non_ex_function")

    with pytest.raises(CantResolveCallable):
        resolve_callable("non_ex_global_func")

    assert resolve_callable("_global_func")() == "global_func_result"
