"""
Command-line interface LLM client implementation.
"""
import asyncio
import shlex
import shutil
import subprocess
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


def run_streaming(argv: list[str], on_chunk, stdin_data: str | None = None) -> str:
    """
    Run the given command line, streaming stdout to ``on_chunk`` as it arrives.

    Plain blocking subprocess - no asyncio. This sidesteps the Windows event
    loop limitation where ``SelectorEventLoop`` (installed globally by Jupyter /
    ipykernel) cannot spawn subprocesses.

    Args:
        argv (list[str]): Command line, already split into arguments.
        on_chunk (callable): Called with each line of stdout as it arrives.
        stdin_data (str | None): If given, written to the process's stdin and
            the pipe is closed. Used to pass the prompt without putting it on the
            command line, which avoids OS argument-length limits (notably the
            Windows ``CreateProcess`` ~32KB limit) for large prompts.
    Returns:
        str: The full stdout once the process completes.
    Raises:
        CommandLineLLMError: If the process exits with a non-zero code, carrying
            stderr as the message if available, otherwise the output.
    """
    with subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        bufsize=1,  # line-buffered
    ) as proc:
        # Feed stdin from a thread so a large prompt can't deadlock against the
        # stdout/stderr read loops by filling the pipe buffer before we read.
        stdin_thread = None
        if stdin_data is not None:
            def _write_stdin():
                try:
                    proc.stdin.write(stdin_data)
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass  # process exited early; stdout/stderr carry the reason

            stdin_thread = threading.Thread(target=_write_stdin)
            stdin_thread.start()

        # Drain stderr in a thread so a full stderr pipe buffer can never deadlock
        # the stdout read loop below.
        stderr_chunks: list[str] = []
        stderr_thread = threading.Thread(
            target=lambda: stderr_chunks.append(proc.stderr.read())
        )
        stderr_thread.start()

        chunks = []
        first = True
        for line in iter(proc.stdout.readline, ""):  # one line per flush, as it arrives
            if first:
                first = False
                if line.startswith(">"):
                    line = line[1:].lstrip()
            chunks.append(line)
            on_chunk(line)

        stderr_thread.join()  # finish reading stderr before the context closes the pipe
        if stdin_thread is not None:
            stdin_thread.join()
        return_code = proc.wait()

    output = "".join(chunks).strip()
    if return_code != 0:
        raise CommandLineLLMError("".join(stderr_chunks).strip() or output)
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
        """Build the argv and callback list shared by the sync/async clients."""
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

        # subprocess passes argv elements directly (no shell), so the prompt is
        # inserted verbatim - shell-quoting it would leak literal quotes into the
        # argument the CLI receives. Splitting happens before substitution, so the
        # prompt always stays within a single argv element.
        argv = shlex.split(self.config.LLM_CLI, posix=True)
        if any(self.PLACEHOLDER in a for a in argv):
            # Placeholder present: substitute the prompt into the command line.
            argv = [a.replace(self.PLACEHOLDER, prompt_str) for a in argv]
            stdin_data = None
        else:
            # No placeholder: pass the prompt via stdin. This avoids OS
            # argument-length limits (notably Windows CreateProcess ~32KB) that
            # would otherwise truncate or reject large prompts, and works with
            # CLIs that read the prompt from stdin (e.g. `claude -p`).
            stdin_data = prompt_str
        resolved = shutil.which(argv[0])
        if resolved:
            argv[0] = resolved
        return argv, callbacks, stdin_data

    def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse | ImageGenerationResponse | StoredImageGenerationResponse:
        """
        Run the command line with the prompt, streaming output to callbacks as it arrives.
        """
        argv, callbacks, stdin_data = self.prepare_run(prompt, kwargs)

        def on_chunk(text):
            for cb in callbacks:
                cb(text)

        result = run_streaming(argv, on_chunk, stdin_data)
        return LLMResponse(
            result,
            api_type=ApiType.CLI,
        )


class AsyncCommandLineClient(BaseAsyncAIClient):
    """
    Asynchronous version of CommandLineClient.
    """
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
        argv, callbacks, stdin_data = self.sync_client.prepare_run(prompt, kwargs)
        loop = asyncio.get_running_loop()

        def on_chunk(text):
            # Runs in the worker thread; marshal coroutine callbacks back to the
            # event loop and block here until they complete.
            for cb in callbacks:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.run_coroutine_threadsafe(cb(text), loop).result()
                else:
                    cb(text)

        result = await asyncio.to_thread(run_streaming, argv, on_chunk, stdin_data)
        return LLMResponse(
            result,
            api_type=ApiType.CLI,
        )
