import logging
import random
import subprocess
import sys
from datetime import datetime

from .file_storage import storage
from .utils import file_link, dedent


def make_silent(code: str) -> str:
    definition = "def dummy_print(*args, **kwargs): ...\n"
    code = code.replace("print(", "dummy_print(")
    if definition not in code:
        code = definition + code
    return code


def execute(
    program: str,
    name=None,
    timeout=15,
    cleanup: bool = True,
    traceback: bool = True,
    log_errors: bool = True,
) -> tuple[str, str]:
    """Executes the provided Python program and returns the output and error messages."""
    program = dedent(program)
    if name is None:
        date = datetime.now().strftime("%Y_%m_%d")
        rnd_int = random.randint(0, sys.maxsize - 1)
        name = f"code/program_{date}__{rnd_int}"
    fn = storage.abs_path(storage.write(f"{name}.py", program, rewrite_existing=False))
    logging.info(f"Executing the program: {file_link(fn)}")
    cmd = [sys.executable, fn]
    stdout, stderr = "", ""
    try:
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                if proc.returncode != 0:
                    log_errors and logging.error(
                        f"Error executing the program, STDERR:\n<\n{stderr}\n>\n",
                    )
            except subprocess.TimeoutExpired:
                proc.kill()
                error = (
                    f"The command timed out (max execution time is {timeout} seconds)"
                )
                stderr = f"{stderr}\n{error}"
                log_errors and logging.error(stderr)
    finally:
        if cleanup:
            try:
                storage.delete(fn)
            except Exception as e:  # pylint: disable=broad-except
                logging.error(f"Error deleting the file: {fn}", e)
        else:
            logging.info(f"Executed code: {file_link(fn)}")

    stdout, stderr = str(stdout).strip(), str(stderr).strip()

    if stderr and not traceback:
        stderr = stderr.splitlines()[-1].lstrip()

    return stdout, stderr
