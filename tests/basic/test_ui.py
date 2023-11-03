from unittest.mock import patch
from microcore.ui import info, debug, error, ask_yn


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
