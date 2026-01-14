import pytest
import microcore as mc
from . import setup


def test_to_json(setup):
    with pytest.raises(mc.BadAIJsonAnswer):
        mc.llm('123{123"field1":"value1"}').parse_json()
    assert mc.llm('123{123"field1":"value1"}').parse_json(raise_errors=False) is False
    assert dict() == mc.llm("{}").parse_json(raise_errors=False)

    mc.llm('{"field1":"value1"}').parse_json(required_fields=["field1"])
    with pytest.raises(mc.BadAIJsonAnswer):
        mc.llm('{"field1":"value1"}').parse_json(required_fields=["field2"])


def test_to_json_and_validate(setup):
    def validator(data: dict):
        assert "field1" in data

    assert mc.llm('{"field1":"value1"}').parse_json(validator=validator)["field1"] == "value1"

    with pytest.raises(mc.BadAIAnswer):
        mc.llm('{"field2":"value1"}').parse_json(validator=validator)
