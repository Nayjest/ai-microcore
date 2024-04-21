from microcore.utils import extract_number


def test_extract_number():
    assert extract_number("123") == 123
    assert extract_number("123.45") == 123.45
    assert extract_number("123.45", dtype=int) == 45
    assert extract_number("123.45", dtype=int, rounding=True) == 123
    assert extract_number("1 2 3") == 3
    assert extract_number("1 2 3", position='first') == 1
