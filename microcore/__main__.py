"""
Command-line entry point for MicroCore.

Usage:
    python -m microcore test-llm [<.env-file>]
"""

import sys
import microcore as mc


def test_llm(env_file: str) -> int:
    """
    Smoke-test the configured LLM: ask for the capital of France
    and verify the answer contains "Paris".
    """
    try:
        mc.configure(
            DOT_ENV_FILE=env_file,
            USE_LOGGING=mc.PRINT_STREAM,
        )
        answer = mc.llm("What is the capital of France?")
        if "paris" not in str(answer).lower():
            raise ValueError('LLM response does not contain expected answer ("Paris").')
    except Exception as e:
        print(mc.ui.red(f"\n[FAIL]: {e}"))
        return 1
    print(mc.ui.green("\n[OK]"))
    return 0


def main(argv: list[str] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    command, *args = argv
    if command == "test-llm":
        if len(args) != 1:
            print(mc.ui.red("test-llm accepts one argument: <.env-file>"))
            return 1
        return test_llm(args[0])
    print(mc.ui.red(f"Unknown command: {command}"))
    print(__doc__.strip())
    return 1


if __name__ == "__main__":
    sys.exit(main())
