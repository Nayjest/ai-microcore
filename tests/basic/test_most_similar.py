from microcore.utils import most_similar

def test_most_similar():
    assert most_similar(
        "hello world",
        ["some str", "hello wrld", "hello there", "hi world"]
    )[0] == "hello wrld"