import asyncio
import microcore.python as py


def test_python_out():
    # language=python
    o, e = py.execute("""
    for i in range(10):
        print(i, end='')
    """)
    assert o, e == ("0123456789", "")


def test_error():
    o, e = py.execute("""
    not a python code
    """, traceback=False, log_errors=False)
    assert o == ""
    assert e == "SyntaxError: invalid syntax"


async def test_python_exec_inline():
    async def fn(a):
        await asyncio.sleep(0.001)
        print("some output")
        return a * 2

    res, out, errors = await py.execute_inline("fn(3)")
    assert (6, "some output", None) == (res, out, errors)