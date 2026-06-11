import os
import asyncio
from contextlib import contextmanager
from unittest.mock import patch

import pytest

import microcore as mc


@contextmanager
def _clean_env():
    """Run a test with a cleared ``os.environ``, restoring it on exit."""
    original_env = dict(os.environ)
    os.environ.clear()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def _azure_entra_base_kwargs() -> dict:
    return dict(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.OPENAI,
        LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
        LLM_API_BASE="https://example.openai.azure.com",
        LLM_API_VERSION="2024-02-15-preview",
        LLM_DEPLOYMENT_ID="dep",
        MODEL="dep",
        LLM_AZURE_USE_ENTRA_ID=True,
    )


def test_valid():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(USE_DOT_ENV=False, LLM_API_KEY="123")
    mc.validate_config()
    assert not mc.env().config.uses_local_model()
    os.environ.update(original_env)


def test_no_key():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(USE_DOT_ENV=False)
    os.environ.update(original_env)


def test_azure_no_deployment():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(
            USE_DOT_ENV=False,
            LLM_API_TYPE=mc.ApiType.OPENAI,
            LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
            LLM_API_KEY="123",
            LLM_API_VERSION="123",
            LLM_API_BASE="https://example.com",
        )
        mc.validate_config()
    os.environ.update(original_env)


def test_azure_ok():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
        LLM_API_KEY="123",
        LLM_DEPLOYMENT_ID="123",
        LLM_API_VERSION="123",
        LLM_API_BASE="https://example.com",
    )

    mc.validate_config()
    assert not mc.env().config.uses_local_model()
    os.environ.update(original_env)


def test_azure_entra_ok_without_api_key():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    try:
        mc.configure(
            USE_DOT_ENV=False,
            LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
            LLM_AZURE_USE_ENTRA_ID=True,
            LLM_DEPLOYMENT_ID="dep",
            LLM_API_VERSION="2024-02-15-preview",
            LLM_API_BASE="https://example.openai.azure.com",
        )
        mc.validate_config()
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_azure_entra_invalid_credential_mode():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    try:
        with pytest.raises(mc.LLMConfigError):
            mc.configure(
                USE_DOT_ENV=False,
                LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
                LLM_AZURE_USE_ENTRA_ID=True,
                LLM_AZURE_ENTRA_CREDENTIAL="wrong",
                LLM_DEPLOYMENT_ID="dep",
                LLM_API_VERSION="2024-02-15-preview",
                LLM_API_BASE="https://example.openai.azure.com",
            )
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_azure_entra_ok_from_process_env():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    try:
        os.environ["LLM_API_TYPE"] = mc.ApiType.OPENAI
        os.environ["LLM_API_PLATFORM"] = mc.ApiPlatform.AZURE
        os.environ["LLM_API_BASE"] = "https://example.openai.azure.com"
        os.environ["LLM_DEPLOYMENT_ID"] = "dep"
        os.environ["LLM_API_VERSION"] = "2024-02-15-preview"
        os.environ["MODEL"] = "dep"
        os.environ["LLM_AZURE_USE_ENTRA_ID"] = "true"
        mc.configure(USE_DOT_ENV=False)
        mc.validate_config()
        assert mc.env().config.LLM_AZURE_ENTRA_SCOPE == "https://ai.azure.com/.default"
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_azure_no_version():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(
            LLM_DEPLOYMENT_ID="123",
            USE_DOT_ENV=False,
            LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
            LLM_API_KEY="123",
            LLM_API_BASE="https://example.com",
        )
        mc.validate_config()

    os.environ.update(original_env)


def test_local_llm():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()

    def inference(prompt, **kwargs):
        return prompt

    mc.configure(
        USE_DOT_ENV=False, LLM_API_TYPE=mc.ApiType.FUNCTION, INFERENCE_FUNC=inference
    )
    assert mc.env().config.uses_local_model()
    assert mc.llm("test") == "test"
    mc.configure(USE_DOT_ENV=False, INFERENCE_FUNC=lambda x: x + ":1")
    assert mc.llm("test") == "test:1"
    assert mc.env().config.uses_local_model()
    with pytest.raises(mc.LLMConfigError):
        mc.configure(
            USE_DOT_ENV=False,
            LLM_API_TYPE=mc.ApiType.FUNCTION,
        )

    os.environ.update(original_env)


def test_none():  # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.NONE,
    )
    assert mc.env().config.uses_local_model()
    with pytest.raises(mc.LLMConfigError):
        mc.llm("test")
    with pytest.raises(mc.LLMConfigError):

        async def fn():
            await asyncio.sleep(0.0001)
            await mc.allm("test")

        asyncio.run(fn())
    os.environ.update(original_env)


# ---------------------------------------------------------------------------
# Azure Entra ID — per-mode credential validation
# ---------------------------------------------------------------------------


def test_azure_entra_client_secret_ok():  # noqa
    """Validation accepts a fully-specified client_secret config."""
    with _clean_env():
        mc.Config(
            **_azure_entra_base_kwargs(),
            LLM_AZURE_ENTRA_CREDENTIAL="client_secret",
            LLM_AZURE_TENANT_ID="tid",
            LLM_AZURE_CLIENT_ID="cid",
            LLM_AZURE_CLIENT_SECRET="sek",
        )


@pytest.mark.parametrize("missing_field", [
    "LLM_AZURE_TENANT_ID",
    "LLM_AZURE_CLIENT_ID",
    "LLM_AZURE_CLIENT_SECRET",
])
def test_azure_entra_client_secret_missing_required(missing_field):  # noqa
    kwargs = dict(
        **_azure_entra_base_kwargs(),
        LLM_AZURE_ENTRA_CREDENTIAL="client_secret",
        LLM_AZURE_TENANT_ID="tid",
        LLM_AZURE_CLIENT_ID="cid",
        LLM_AZURE_CLIENT_SECRET="sek",
    )
    kwargs[missing_field] = ""
    with _clean_env(), pytest.raises(mc.LLMConfigError):
        mc.Config(**kwargs)


@pytest.mark.parametrize("mode", ["managed_identity", "certificate", "workload_identity"])
def test_azure_entra_non_supported_modes_fail_validation(mode):  # noqa
    with _clean_env(), pytest.raises(mc.LLMConfigError):
        mc.Config(
            **_azure_entra_base_kwargs(),
            LLM_AZURE_ENTRA_CREDENTIAL=mode,
            LLM_AZURE_TENANT_ID="tid",
            LLM_AZURE_CLIENT_ID="cid",
            LLM_AZURE_CLIENT_SECRET="sek",
        )


def test_build_azure_entra_token_provider_client_secret_uses_config_values():  # noqa
    """ClientSecretCredential is constructed from LLM_AZURE_* fields only."""
    from microcore.llm import openai as openai_module

    fake_provider = lambda: "tok"  # noqa: E731
    with _clean_env(), \
         patch("azure.identity.get_bearer_token_provider", return_value=fake_provider), \
         patch("azure.identity.ClientSecretCredential") as csc:
        csc.return_value = object()
        cfg = mc.Config(
            **_azure_entra_base_kwargs(),
            LLM_AZURE_ENTRA_CREDENTIAL="client_secret",
            LLM_AZURE_TENANT_ID="tid-1",
            LLM_AZURE_CLIENT_ID="cid-1",
            LLM_AZURE_CLIENT_SECRET="secret-1",
        )
        openai_module._build_azure_entra_token_provider(cfg)
        csc.assert_called_once_with(
            tenant_id="tid-1",
            client_id="cid-1",
            client_secret="secret-1",
        )


def test_describe_masks_entra_secrets():  # noqa
    """``Config.describe()`` must redact secret-like fields."""
    with _clean_env():
        cfg = mc.Config(
            **_azure_entra_base_kwargs(),
            LLM_AZURE_ENTRA_CREDENTIAL="client_secret",
            LLM_AZURE_TENANT_ID="tid",
            LLM_AZURE_CLIENT_ID="cid",
            LLM_AZURE_CLIENT_SECRET="super-secret-value-1234567890",
        )
        described = cfg.describe(return_dict=True)
        for key in ("azure_client_secret",):
            assert key in described, f"missing {key!r} in describe(): {described}"
            assert "****" in described[key] or described[key] == "***", (
                f"{key} not masked: {described[key]!r}"
            )
        # Non-sensitive fields are NOT masked
        assert described.get("azure_tenant_id") == "tid"
        assert described.get("azure_client_id") == "cid"
