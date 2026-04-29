"""
Smoke test: Azure OpenAI / Foundry with Microsoft Entra ID.

Configuration is read from environment variables (or local `.env` loaded by
MicroCore). See `.env.example` for all relevant fields.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import dotenv

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Load local .env early so required-field checks can read values from os.environ.
dotenv.load_dotenv(_root / ".env", override=True)

import microcore as mc


def _common_kwargs() -> dict:
    llm_api_base = (os.getenv("LLM_API_BASE") or "").rstrip("/").strip()
    llm_deployment_id = (os.getenv("LLM_DEPLOYMENT_ID") or "").strip()
    llm_api_version = (
        os.getenv("LLM_API_VERSION") or "2024-12-01-preview"
    ).strip()
    return dict(
        USE_DOT_ENV=True,
        DOT_ENV_FILE=_root / ".env",
        LLM_API_TYPE=mc.ApiType.OPENAI,
        LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
        LLM_API_BASE=llm_api_base,
        LLM_API_VERSION=llm_api_version,
        LLM_DEPLOYMENT_ID=llm_deployment_id,
        MODEL=llm_deployment_id,
        LLM_AZURE_USE_ENTRA_ID=True,
        VALIDATE_CONFIG=True,
        EMBEDDING_DB_TYPE=mc.EmbeddingDbType.NONE,
    )


def _require_env(var_name: str) -> str:
    value = (os.getenv(var_name) or "").strip()
    if not value:
        sys.exit(f"Missing required environment variable: {var_name}")
    return value


def main() -> None:
    _ = _require_env("LLM_API_BASE")
    deployment_id = _require_env("LLM_DEPLOYMENT_ID")
    mode = (os.getenv("LLM_AZURE_ENTRA_CREDENTIAL") or "client_secret").strip().lower()
    if mode == "client_secret":
        tenant_id = _require_env("LLM_AZURE_TENANT_ID")
        client_id = _require_env("LLM_AZURE_CLIENT_ID")
        client_secret = _require_env("LLM_AZURE_CLIENT_SECRET")
        mc.configure(
            **_common_kwargs(),
            LLM_AZURE_ENTRA_CREDENTIAL="client_secret",
            LLM_AZURE_TENANT_ID=tenant_id,
            LLM_AZURE_CLIENT_ID=client_id,
            LLM_AZURE_CLIENT_SECRET=client_secret,
        )
    else:
        sys.exit(
            "Unsupported LLM_AZURE_ENTRA_CREDENTIAL="
            f"{mode!r}. Use 'client_secret'."
        )
    print(f"[mode={mode}] deployment={deployment_id}")
    print(mc.llm("Reply with exactly: OK"))


if __name__ == "__main__":
    main()
