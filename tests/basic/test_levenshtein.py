from microcore.utils import levenshtein

def test_levenshtein():
    assert levenshtein("", "") == 0
    assert levenshtein("word", "") == 4
    assert levenshtein("", "word") == 4
    assert levenshtein("graph", "giraffe") == 4
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("flaw", "lawn") == 2
    assert levenshtein("intention", "execution") == 5
    assert levenshtein("abc", "def") == 3
    assert levenshtein("abc", "cde") == 3
    assert levenshtein("abc", "a_b_c") == 2
    assert levenshtein("abc", "\nabc \n") == 3
    assert levenshtein("abc", "abc") == 0