from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import microcore as mc

# Not secret: replace with your resource for a local run (do not commit real values in a PR).
LLM_API_BASE = "https://YOUR_RESOURCE.openai.azure.com/"
LLM_DEPLOYMENT_ID = "YOUR_DEPLOYMENT"
LLM_API_VERSION = "2024-02-15-preview"


def main() -> None:
    """
    Smoke test: Azure OpenAI / Foundry with Entra (no API key).

    Endpoint, deployment, and API version are set in this file (placeholders). Values listed in
    ``.env.example`` (``LLM_API_KEY``, optional ``LLM_AZURE_ENTRA_SCOPE`` / ``LLM_AZURE_ENTRA_CREDENTIAL``,
    plus the usual app keys) load from your **local** ``.env`` via ``USE_DOT_ENV``. Do not commit ``.env``.
    """
    base = LLM_API_BASE.rstrip("/").strip()
    dep = LLM_DEPLOYMENT_ID.strip()
    ver = LLM_API_VERSION.strip()
    if "YOUR_RESOURCE" in base or dep == "YOUR_DEPLOYMENT":
        sys.exit(
            "Edit LLM_API_BASE / LLM_DEPLOYMENT_ID / LLM_API_VERSION in examples/azure_entra_smoke.py "
            "(placeholders), or copy the script and adjust locally without committing."
        )

    mc.configure(
        USE_DOT_ENV=True,
        DOT_ENV_FILE=_root / ".env",
        LLM_API_TYPE=mc.ApiType.OPENAI,
        LLM_API_PLATFORM=mc.ApiPlatform.AZURE,
        LLM_API_BASE=base,
        LLM_API_VERSION=ver,
        LLM_DEPLOYMENT_ID=dep,
        MODEL=dep,
        LLM_AZURE_USE_ENTRA_ID=True,
        VALIDATE_CONFIG=True,
        EMBEDDING_DB_TYPE=mc.EmbeddingDbType.NONE,
    )
    print(mc.llm("Reply with exactly: OK"))


if __name__ == "__main__":
    main()
