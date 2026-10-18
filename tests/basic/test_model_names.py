import pytest
import microcore as mc


def test_model_names_raises_when_default_client_is_missing(monkeypatch):
    class FakeEnv:
        default_client = None

    monkeypatch.setattr(mc, "env", lambda: FakeEnv())

    with pytest.raises(ValueError, match="No default LLM client is configured"):
        mc.model_names()


def test_model_names_returns_names_from_default_client(monkeypatch):
    class FakeClient:
        @staticmethod
        def model_names():
            return ["model-a", "model-b"]

    class FakeEnv:
        default_client = FakeClient()

    monkeypatch.setattr(mc, "env", lambda: FakeEnv())

    assert mc.model_names() == ["model-a", "model-b"]
