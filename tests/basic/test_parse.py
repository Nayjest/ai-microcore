import microcore as mc


def test_parse():
    document = (
        "[[section1]]\n"
        "section 1 line 1\n"
        "[[section2]]\n"
        "section 2 line 1\n"
        "section 2 line 2\n"
    )
    d = mc.parse(document)
    assert 'section1' in d
    assert 'section2' in d
    assert d['section1'] == "section 1 line 1"
    assert d['section2'] == "section 2 line 1\nsection 2 line 2"


def test_parse_ln():
    document = (
        "[[section1]]\n"
        "section 1 line 1\n"
    )
    d = mc.parse(document)
    assert 'section1' in d
    assert d['section1'] == "section 1 line 1"


def test_parse_no_end():
    document = (
        "[[section1]]\n"
        "section 1 line 1\n"
        "[[end]]"
    )
    d = mc.parse(document)
    assert 'end' not in d


def test_parse_format():
    document = (
        "@section_1\n"
        "section 1 line 1\n"
        "section 1 line 2\n"
        "@end"
    )
    d = mc.parse(document, field_format=r'\@(.*?)')
    assert d['section_1'] == "section 1 line 1\nsection 1 line 2"
