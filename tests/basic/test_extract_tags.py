from textwrap import dedent
from microcore.utils import extract_tags


def test_extract_args():
    text = dedent(
        """
    Here is some text with <tag1>value1</tag1> and <tag2>value2</tag2>.
    Also, there is a <tag3>
    multi-line
    value3
    </tag3>.
    <tag4 attr1="value1" attr2=value2 attr3=3 a4='' p ="5" a6 = some_val@!.-#$%>content</tag4>
    <brokentag>no end tag
    <otherbroken attr=noend>more text</otherbroken3>
    """
    )
    expected = [
        ("tag1", {}, "value1"),
        ("tag2", {}, "value2"),
        ("tag3", {}, "multi-line\nvalue3"),
        (
            "tag4",
            {
                "attr1": "value1",
                "attr2": "value2",
                "attr3": "3",
                "a4": "",
                "p": "5",
                "a6": "some_val@!.-#$%",
            },
            "content",
        ),
    ]
    assert extract_tags(text, strip=True) == expected


def test_extract_ignore_inner():
    text = dedent(
        """
    <tag1>
    <tag2>value</tag2>
    </tag1>
    <tag3 attr="val"></tag3>
    """
    )
    expected = [
        ("tag1", {}, "<tag2>value</tag2>"),
        ("tag3", {"attr": "val"}, ""),
    ]
    assert extract_tags(text, strip=True) == expected
