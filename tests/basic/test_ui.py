import os
import subprocess
import sys
from unittest.mock import patch

from microcore.ui import info, debug, error, ask_yn, colors_disabled

_COLOR_SCRIPT = (
    "import logging, microcore.ui as ui;"
    "print(ui.red('red text'));"
    "ui.error('error text');"
    "logging.basicConfig();"
    "logging.getLogger().warning(ui.yellow('log text'))"
)


def _run_color_script(no_color: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
    if no_color is not None:
        env["NO_COLOR"] = no_color
    result = subprocess.run(
        [sys.executable, "-c", _COLOR_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout + result.stderr


def test_colors_disabled_flag():
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert colors_disabled() is True
    with patch.dict(os.environ, {"NO_COLOR": ""}):
        assert colors_disabled() is False


def test_no_color_strips_ansi_codes():
    output = _run_color_script("1")
    assert "red text" in output
    assert "error text" in output
    assert "log text" in output
    assert "\x1b[" not in output


def test_colors_kept_without_no_color():
    output = _run_color_script(None)
    assert "\x1b[" in output


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
