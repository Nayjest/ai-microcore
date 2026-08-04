# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, etc.) when working with
code in this repository.

## Overview

AI MicroCore (`ai-microcore`, importable as `microcore`) is a thin adapter layer over LLM
providers and vector/semantic-search databases. The whole point is provider-agnosticism:
application code calls `llm(...)`, `tpl(...)`, `texts.search(...)` and the concrete backend
(OpenAI, Anthropic, Google, a CLI tool, a local Transformers model, an arbitrary Python
function, ChromaDB, Qdrant, ...) is selected purely by configuration. When changing behavior,
preserve this property — adding a parameter or branch that forces application code to know which
backend is active works against the core design.

## Commands

Tests run locally with `pytest` (no Docker required); the `Makefile` targets wrap the same
commands in a Docker container (`make test`, `make cs`, `make black`) and are mainly for the
maintainer's workflow.

```bash
pip install -r requirements/dev.txt   # test + lint + build tooling

pytest                                 # runs tests/basic (the default testpath, no network)
pytest tests/basic/test_llm.py         # single file
pytest tests/basic/test_llm.py::test_name   # single test
pytest tests/apis                      # hits REAL API services; needs .env.test.* + keys

flake8 microcore tests examples        # max-line-length=100 (.flake8)
pylint microcore                       # config in .pylintrc
black microcore tests examples         # formatter; commits are expected to be black-clean

python -m microcore test-llm <.env-file> [<prompt>]   # smoke-test a provider config live
```

Note `pyproject.toml` sets `testpaths = ["tests/basic"]` and `asyncio_mode = "auto"`, so a bare
`pytest` only runs the basic suite. Other suites: `tests/extended`, `tests/cache`, `tests/local`
(Transformers), `tests/apis` (live). CI (`.github/workflows/tests.yml`) runs the basic suite on
Python 3.10–3.13 and pylint separately.

## Architecture

**Configuration → Env → functions.** Everything flows from a single `Config` dataclass
(`microcore/configuration.py`) into a global `Env` (`microcore/_env.py`). `configure(**params)`
builds the `Env`, which wires up the templating engine, the LLM functions, and the embedding DB.
The module-level helpers (`llm`, `allm`, `tpl`, `texts`, etc.) all read from this global env.
`Config.__post_init__` does substantial normalization: it resolves `LLM_API_TYPE` /
`LLM_API_PLATFORM` from many inputs (deprecated aliases, platform names, `LLM_API_BASE` sniffing,
presence of `LLM_CLI` or `INFERENCE_FUNC`), so the same target can be reached several ways.
Config precedence is unusual: explicit `configure(...)` arguments win, then the `.env` file
(`DOT_ENV_FILE`), and OS environment variables have the *lowest* priority.

**Backend selection.** `ApiType` and `ApiPlatform` enums live in `microcore/llm_backends.py`.
`ApiType` is the transport family (OPENAI, ANTHROPIC, GOOGLE, TRANSFORMERS, CLI, FUNCTION, ...);
`ApiPlatform` is a concrete vendor (MISTRAL, XAI, GROQ, DEEPSEEK, ...) that maps onto an
`ApiType` and a default API base. Many vendors are reached via the OpenAI-compatible transport,
so they are platforms over `ApiType.OPENAI` rather than new api types. The actual client
implementations are in `microcore/llm/` (`openai.py`, `anthropic.py`, `google_genai.py`,
`cli.py`, `local_llm.py`, `local_transformers.py`, `shared.py`). The OpenAI transport can also
speak the Responses API (`LLM_USE_RESPONSES_API=true`, implemented in `azure_responses.py`) —
used for Azure endpoints and models that require it.

**LLM call path.** `microcore/_llm_functions.py` exposes `llm`/`allm`/`llm_parallel`/
`llm_stream`. Prompt arguments are normalized in `_prepare_llm_args.py` (strings, lists, dicts,
and `Msg`/`SysMsg`/`UserMsg`/`AssistantMsg` from `message_types.py` into provider message
format). Responses are wrapped by `wrappers/llm_response_wrapper.py` (`LLMResponse`) — a `str`
subclass that also carries `.choices` and other response fields, so the return value is a string
but exposes full metadata. `env().llm_before_handlers` / `llm_after_handlers` are hook lists for
cross-cutting behavior (logging, metrics, caching).

**Templating.** `tpl(file, **params)` renders Jinja2 templates (`microcore/templating/`) from
`PROMPT_TEMPLATES_PATH` (default `./tpl`); `prompt(str, **params)` (alias `fmt`) renders from a
string. Both return a `PromptWrapper` (str subclass) carrying the template vars/file.

**Vector DB.** `microcore.texts` is an `AbstractEmbeddingDB` (`microcore/embedding_db/`) with
`chromadb.py` (default) and `qdrant.py` implementations, selected via `EMBEDDING_DB_TYPE`.

**MCP.** `microcore/mcp.py` + `MCPRegistry` connect MCP tool servers to any LLM backend,
including providers that don't natively support MCP. `microcore/ai_func/` turns plain Python
functions into LLM tool definitions via `@ai_func` (rendered through the `ai-func.*.j2` templates).

## Conventions

- Public API is whatever `microcore/__init__.py` re-exports; treat that file as the surface and
  keep new public symbols flowing through it. Modules prefixed with `_` (`_env`, `_llm_functions`,
  `_prepare_llm_args`) and `llm/__init__.py` (marked `@private`) are internal.
- Several `ApiType`/`ApiPlatform` members are marked `@Deprecated` in favor of `LLM_API_PLATFORM`;
  prefer the platform parameter over adding new `ApiType` values for OpenAI-compatible vendors.
- `.env.*.example` files document the configuration shape for each provider — update the relevant
  one (and the README) when adding or changing configuration options.
- Supports Python 3.10–3.14; avoid syntax newer than 3.10.
