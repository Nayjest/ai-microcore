from . import *  # noqa


def test_to_json(setup):
    with pytest.raises(mc.BadAIJsonAnswer):
        mc.llm('123{"field1":"value1"}').parse_json()
    assert False == mc.llm('123{"field1":"value1"}').parse_json(raise_errors=False)
    assert dict() == mc.llm("{}").parse_json(raise_errors=False)

    mc.llm('{"field1":"value1"}').parse_json(required_fields=["field1"])
    with pytest.raises(mc.BadAIJsonAnswer):
        mc.llm('{"field1":"value1"}').parse_json(required_fields=["field2"])
