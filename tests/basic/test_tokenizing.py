import logging

import microcore.tokenizing as tokenizing


def test_unresolved_model_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(tokenizing.tiktoken, "get_encoding", lambda name: name)
    monkeypatch.setattr(tokenizing, "_unresolved_models", set())
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            tokenizing._resolve_tiktoken_encoding(for_model="claude-opus-4-5")
    warnings = [r for r in caplog.records if "claude-opus-4-5" in r.getMessage()]
    assert len(warnings) == 1
