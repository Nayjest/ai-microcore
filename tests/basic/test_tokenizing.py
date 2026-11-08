from microcore import tokenizing


def test_unknown_model_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(tokenizing, "_unresolved_models", set())
    for _ in range(3):
        tokenizing.num_tokens_from_string("hi", for_model="claude-opus-4-5")
    assert caplog.text.count("claude-opus-4-5") == 1
