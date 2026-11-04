import logging
from unittest.mock import MagicMock, patch

from microcore import tokenizing


def _resolve_unknown(model: str):
    with patch.object(
        tokenizing.tiktoken, "encoding_for_model", side_effect=KeyError(model)
    ), patch.object(tokenizing.tiktoken, "get_encoding", return_value=MagicMock()):
        return tokenizing._resolve_tiktoken_encoding(for_model=model)


def test_unresolved_model_warns_only_once(caplog):
    tokenizing._unresolved_models.discard("unknown-model-x")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert _resolve_unknown("unknown-model-x") is not None
    warnings = [r for r in caplog.records if "unknown-model-x" in r.getMessage()]
    assert len(warnings) == 1


def test_unresolved_warning_is_per_model(caplog):
    tokenizing._unresolved_models.discard("unknown-model-a")
    tokenizing._unresolved_models.discard("unknown-model-b")
    with caplog.at_level(logging.WARNING):
        _resolve_unknown("unknown-model-a")
        _resolve_unknown("unknown-model-b")
        _resolve_unknown("unknown-model-a")
    messages = [r.getMessage() for r in caplog.records]
    assert sum("unknown-model-a" in m for m in messages) == 1
    assert sum("unknown-model-b" in m for m in messages) == 1
