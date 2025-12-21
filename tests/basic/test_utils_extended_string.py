import pytest
from microcore.utils import ExtendedString


def test_extended_string_basic():
    s = ExtendedString(
        "The price is 123.45 USD",
        attrs=dict(payload_attribute=123.45)
    )
    # Check payload attribute
    assert isinstance(s, ExtendedString)
    assert hasattr(s, "payload_attribute")
    assert s.payload_attribute == 123.45
    assert getattr(s, "payload_attribute") == 123.45
    assert s.__dict__ == {"payload_attribute": 123.45}
    s.payload_attribute = 125
    assert s.payload_attribute == 125

    # Test behave like a string
    assert s == "The price is 123.45 USD"
    assert isinstance(s, str)
    assert s.upper() == "THE PRICE IS 123.45 USD"
    assert s.split() == ["The", "price", "is", "123.45", "USD"]
    assert s[0] == "T"

    assert s.replace("USD", "EUR") == "The price is 123.45 EUR"
    with pytest.raises(AttributeError):
        _ = s[:3].payload_attribute

    assert "The" == s[:3]
    with pytest.raises(AttributeError):
        _ = s[:3].payload_attribute


def test_no_attrs():
    s = ExtendedString("simple string")
    assert s == "simple string"
    with pytest.raises(AttributeError):
        _ = s.some_attribute
    assert not hasattr(s, "some_attribute")

def test_extended_string_alternative_constructors():
    assert ExtendedString(string="my_text", attrs=dict(myattr="myval")).myattr == "myval"
    assert ExtendedString("my_text", dict(myattr="myval")).myattr == "myval"
    assert ExtendedString("my_text", myattr="myval").myattr == "myval"

