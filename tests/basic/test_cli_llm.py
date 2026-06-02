"""
Tests for the ApiType.CLI backend (microcore/llm/cli.py).

The "mock" here is a tiny parrot script invoked as a real subprocess - the most
honest stand-in for a CLI LLM, since the backend genuinely spawns an external
process and streams its stdout back.
"""
import pytest

import microcore as mc
from microcore.llm.cli import CommandLineLLMError

# A minimal CLI "LLM": echoes the request back, fails on "boom", or - on "stream3" -
# emits three words one at a time (flushed, with a pause) to exercise streaming.
PARROT = """\
import sys, time
req = sys.argv[1] if len(sys.argv) > 1 else ""
if req == "boom":
    sys.stderr.write("kaboom")
    sys.exit(1)
if req == "stream3":
    for word in ("alpha", "beta", "gamma"):
        print(word, flush=True)
        time.sleep(0.05)
    sys.exit(0)
print(req)
"""


@pytest.fixture()
def cli_setup(tmp_path):
    parrot = tmp_path / "parrot.py"
    parrot.write_text(PARROT)
    # `python` resolves via shutil.which(); as_posix() keeps the path
    # backslash-free so shlex.split(posix=True) does not mangle it.
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.CLI,
        LLM_CLI=f"python {parrot.as_posix()} <request>",
        MODEL="parrot-cli",
    )
    yield


def test_cli_llm_sync(cli_setup):
    res = mc.llm("ping")
    assert res == "ping"
    assert res.api_type == mc.ApiType.CLI


def test_cli_llm_multiword_prompt(cli_setup):
    # The whole prompt must reach the CLI as a single argv element.
    assert mc.llm("hello world") == "hello world"


async def test_cli_llm_async(cli_setup):
    assert await mc.allm("pong") == "pong"


def test_cli_llm_streaming_callback(cli_setup):
    chunks = []
    res = mc.llm("stream me", callback=chunks.append)
    assert "".join(chunks).strip() == "stream me"
    assert res == "stream me"


def test_cli_llm_streams_separate_chunks(cli_setup):
    # Three words emitted one-at-a-time should reach the callback as three
    # separate chunks, not a single coalesced blob.
    chunks = []
    res = mc.llm("stream3", callback=chunks.append)
    assert [c.strip() for c in chunks] == ["alpha", "beta", "gamma"]
    assert res.split() == ["alpha", "beta", "gamma"]


async def test_cli_llm_async_callback(cli_setup):
    chunks = []

    async def collect(text):
        chunks.append(text)

    res = await mc.allm("async stream", callback=collect)
    assert "".join(chunks).strip() == "async stream"
    assert res == "async stream"


def test_cli_llm_error(cli_setup):
    with pytest.raises(CommandLineLLMError) as exc_info:
        mc.llm("boom")
    # stderr of the failed process is surfaced as the error message
    assert "kaboom" in str(exc_info.value)


def test_cli_model_names(cli_setup):
    assert mc.model_names() == ["parrot-cli"]
