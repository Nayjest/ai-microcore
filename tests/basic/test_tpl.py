from microcore import tpl, configure


def test_tpl(monkeypatch):
    configure(PROMPT_TEMPLATES_PATH='tests/basic/tpl')
    assert tpl('test.j2', var='val') == 'Test template val'
