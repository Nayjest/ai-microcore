import pytest

from microcore import BadAIJsonAnswer
from microcore.json_parsing import unwrap_json_substring, parse_json


def test_unwrap():
    cases = [
        ('{"a": 1}', '{"a": 1}'),
        ('```json{"a": 1}```', '{"a": 1}'),
        (' ```json\n{"a": 1}\n```  ', '{"a": 1}'),
        ('surrounding text\n\nmoretext```json]\n{"a": 1}\n```text_after', '{"a": 1}'),
        (" [] ", "[]"),
        ("not json", "not json"),
    ]
    for dirty, expected in cases:
        assert unwrap_json_substring(dirty) == expected

    assert (
        unwrap_json_substring(
            ' qq ```json{"a": 1}```',
            allow_in_text=False,
            return_original_on_fail=False,
        )
        == ""
    )

    assert unwrap_json_substring("not json", return_original_on_fail=False) == ""


def test_parse():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('{"a": "value') == {"a": "value"}
    assert parse_json('{"a": "value"') == {"a": "value"}
    assert parse_json("23.4") == 23.4
    parsed = parse_json(
        """

```json
{
    // comment
    "a": 1,
    // comment 2
    "b": 2
    ...
}
```

    """
    )
    assert parsed == {"a": 1, "b": 2}
    assert parse_json('{"a": 1, "b": True}') == {"a": 1, "b": True}
    assert parse_json('{"a": 1, "b":False, "c":"c"}') == {"a": 1, "b": False, "c": "c"}
    assert parse_json('{"field": None}') == parse_json('{"field": null}')
    assert parse_json('{"field": [/* comment */]}') == {"field": []}
    assert not parse_json("not json", raise_errors=False)
    assert parse_json('-- ["123"] -- ', raise_errors=False) == ["123"]

    with pytest.raises(BadAIJsonAnswer):
        parse_json("qq{")
    with pytest.raises(BadAIJsonAnswer):
        parse_json("qwe{}}")
    with pytest.raises(BadAIJsonAnswer):
        parse_json("")
    with pytest.raises(BadAIJsonAnswer):
        parse_json("}{")
