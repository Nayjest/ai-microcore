from microcore import ApiType


def test_in():  # test str in SafeStrEnum for python < 3.12
    assert "openai" in ApiType
