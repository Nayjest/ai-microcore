import microcore as mc
from . import setup


async def test_wrappers(setup):
    out = mc.tpl(file="json_data.j2", var="test_data").to_llm().parse_json()
    assert out == dict(data="test_data")

    out = (await mc.tpl(file="json_data.j2", var="test_data").to_allm()).parse_json()
    assert out == dict(data="test_data")
