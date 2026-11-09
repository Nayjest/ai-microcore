import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from microcore.ui import info, debug, error, ask_yn


# An empty NO_COLOR means "not set" (https://no-color.org)
@pytest.mark.parametrize("no_color, has_colors", [("1", False), ("", True)])
def test_no_color(no_color, has_colors):
    script = (
        "import logging, microcore.ui as ui;"
        "print(ui.red('red text'));"
        "ui.error('error text');"
        "logging.basicConfig();"
        "logging.getLogger().warning(ui.yellow('log text'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "NO_COLOR": no_color},
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout + result.stderr
    assert all(text in output for text in ("red text", "error text", "log text"))
    assert ("\x1b[" in output) == has_colors


def test_info():
    with patch("builtins.print") as mock_print:
        info("Test message")
        args, _ = mock_print.call_args
        assert "Test message" in args[0]


def test_debug():
    with patch("builtins.print") as mock_print:
        debug("Debug message")
        args, _ = mock_print.call_args
        assert "Debug message" in args[0]


def test_error():
    with patch("builtins.print") as mock_print:
        error("Error message")
        args, _ = mock_print.call_args
        assert "Error message" in args[0]


def test_ask_yn_yes():
    with patch("builtins.input", return_value="y"):
        assert ask_yn("Confirm?") is True


def test_ask_yn_no():
    with patch("builtins.input", return_value="n"):
        assert ask_yn("Confirm?") is False
