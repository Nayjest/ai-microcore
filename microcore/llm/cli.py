"""
Command-line interface LLM client implementation.
"""
import asyncio
import shlex
import shutil
import sys
import threading

from ..lm_client import BaseAIChatClient, BaseAsyncAIClient
from ..configuration import Config
from ..llm_backends import ApiType
from ..types import BadAIAnswer, TPrompt
from ..wrappers.llm_response_wrapper import (
    LLMResponse,
    ImageGenerationResponse,
    StoredImageGenerationResponse
)
from .shared import prepare_callbacks


class CommandLineLLMError(BadAIAnswer):
    def __init__(
        self, message: str = "Error executing LLM CLI command", details=None
    ):
        super().__init__(message, details)


def _make_subprocess_loop() -> asyncio.AbstractEventLoop:
    """
    Create an event loop that is able to spawn subprocesses on the current platform.

    On Windows only the ``ProactorEventLoop`` supports subprocesses; the
    ``SelectorEventLoop`` raises ``NotImplementedError`` from
    ``_make_subprocess_transport``. The selector loop is installed globally by
    some libraries (e.g. Jupyter / ipykernel, which need it for pyzmq/tornado),
    so we create the right loop explicitly instead of relying on the ambient
    event loop policy. The global policy is left untouched.
    """
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    return asyncio.new_event_loop()


def _loop_supports_subprocess(loop: asyncio.AbstractEventLoop) -> bool:
    """Whether the given (running) loop is able to spawn subprocesses."""
    if sys.platform != "win32":
        return True
    return isinstance(loop, asyncio.ProactorEventLoop)


def _run_on_new_loop(coro):
    """Run a coroutine to completion on a fresh, subprocess-capable event loop."""
    loop = _make_subprocess_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def _run_sync(coro):
    """
    Run a coroutine to completion from sync code, on a subprocess-capable event
    loop, whether or not a loop is already running in this thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_on_new_loop(coro)  # no loop running here: run directly

    # A loop is already running in this thread; we cannot nest run_until_complete.
    # Run on a fresh loop in another thread and propagate the result or error.
    box = {}

    def runner():
        try:
            box["value"] = _run_on_new_loop(coro)
        except BaseException as e:  # noqa: BLE001 - re-raised in the calling thread
            box["error"] = e

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


async def run_streaming_safe(argv: list[str], callback) -> str:
    """
    Like :func:`run_streaming`, but resilient to event loops that cannot spawn
    subprocesses (e.g. the Windows ``SelectorEventLoop`` used inside Jupyter).

    If the running loop supports subprocesses, the command is run on it directly
    so callbacks share the caller's loop. Otherwise the work is offloaded to a
    subprocess-capable loop in a worker thread.
    """
    loop = asyncio.get_running_loop()
    if _loop_supports_subprocess(loop):
        return await run_streaming(argv, callback)
    return await asyncio.to_thread(_run_on_new_loop, run_streaming(argv, callback))


async def run_streaming(argv: list[str], callback) -> str:
    """
    Run the given command line, streaming stdout to the callback as it arrives.
    Args:
        argv (list[str]): List of command line arguments.
        callback: Async function to call with each chunk of output as it arrives.
    Returns:
        str: The full output as a string once the process completes.
    Raises:
        CommandLineLLMError: If the process exits with a non-zero code, with stderr as
            the error message if available, otherwise the output.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    chunks = []
    first = True

    async def pump_stdout():
        nonlocal first
        async for raw in proc.stdout:        # yields complete lines as they arrive
            text = raw.decode()
            if first:
                first = False
                if text.startswith(">"):
                    text = text[1:].lstrip()
            chunks.append(text)
            await callback(text)

    stderr_data = bytearray()

    async def pump_stderr():
        async for raw in proc.stderr:
            stderr_data.extend(raw)

    await asyncio.gather(pump_stdout(), pump_stderr())
    return_code = await proc.wait()
    output = "".join(chunks).strip()
    if return_code != 0:
        raise CommandLineLLMError(stderr_data.decode().strip() or output)
    return output


class CommandLineClient(BaseAIChatClient):
    """
    LLM client that runs a command-line process for each request,
    streaming output as it arrives.
    """

    # String in the command line to be replaced with the prompt, if present
    PLACEHOLDER = "<request>"

    aio: "AsyncCommandLineClient"

    def __init__(self, config: Config):
        """Initialize the CommandLineClient with the given configuration."""
        super().__init__(config)
        self.aio = AsyncCommandLineClient(self)

    def load_models(self, **_kwargs) -> dict:
        """
        Returns a dict of available models.
        For CLI, we assume the model is specified in the config and is always available.
        """
        return {
            self.config.MODEL: {
                "id": self.config.MODEL,
                "supports_functions": False,
                "supports_vision": False,
            }
        }

    def prepare_run(self, prompt: TPrompt, kwargs: dict):
        """Build the command line and streaming callback shared by sync/async."""
        args = {**self.config.LLM_DEFAULT_ARGS, **kwargs}
        args["model"] = args.get("model", self.config.MODEL)
        callbacks = prepare_callbacks(self.config, args)
        messages = self.convert_prompt_to_chat_input(prompt)

        if len(messages) > 1:
            prompt_str = ""
            for msg in messages:
                prompt_str += f"\n[{msg['role']}]: {msg['content'][0]}"
        else:
            prompt_str = messages[0]['content'][0]

        async def callback(text):
            for cb in callbacks:
                if asyncio.iscoroutinefunction(cb):
                    await cb(text)
                else:
                    cb(text)

        # create_subprocess_exec passes argv elements directly (no shell), so the
        # prompt is inserted verbatim - shell-quoting it would leak literal quotes
        # into the argument the CLI receives.
        argv = shlex.split(self.config.LLM_CLI, posix=True)
        argv = [a.replace(self.PLACEHOLDER, prompt_str) for a in argv]
        resolved = shutil.which(argv[0])
        if resolved:
            argv[0] = resolved
        return argv, callback

    def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse | ImageGenerationResponse | StoredImageGenerationResponse:
        """
        Run the command line with the prompt, streaming output to callbacks as it arrives.
        """
        argv, callback = self.prepare_run(prompt, kwargs)
        result: str = _run_sync(run_streaming(argv, callback))
        return LLMResponse(
            result,
            api_type=ApiType.CLI,
        )


class AsyncCommandLineClient(BaseAsyncAIClient):
    """
    Asynchronous version of CommandLineClient."""
    sync_client: "CommandLineClient"

    def __init__(self, sync_client: "CommandLineClient"):
        """Initialize the AsyncCommandLineClient with a reference to the sync client."""
        self.sync_client = sync_client

    async def load_models(self, **kwargs) -> dict:
        """For CLI, we assume the model is specified in the config and is always available."""
        return self.sync_client.load_models(**kwargs)

    async def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse:
        """Run the command line with the prompt, streaming output to callbacks as it arrives."""
        argv, callback = self.sync_client.prepare_run(prompt, kwargs)
        result: str = await run_streaming_safe(argv, callback)
        return LLMResponse(
            result,
            api_type=ApiType.CLI,
        )
