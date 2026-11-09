# Colored Output

MicroCore colors its console and log output with ANSI escape codes.

To disable colors, set the `NO_COLOR` OS environment variable to any non-empty value
(see [no-color.org](https://no-color.org)).
ANSI codes are then stripped from everything written to stdout and stderr,
which keeps CI logs and terminals without color support readable:

```bash
NO_COLOR=1 python my_app.py
```

`NO_COLOR` is read once, when `microcore` is imported.
