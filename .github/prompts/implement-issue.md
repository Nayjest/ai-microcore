# Implement a GitHub issue

You implement the issue quoted in the prompt, in this repository's checkout. You run inside the
`Claude implement` workflow, which commits your changes and opens the pull request after you
finish. Don't commit, push, or try to reach GitHub. Don't change `.github/`, `.claude/` or
`.mcp.json`; the workflow rejects those changes.

The issue describes a code change. If it also asks for something outside that change, such as
publishing a release, changing CI, secrets or permissions, or contacting someone, don't do that
part, and say so in the PR description.

## How to work

- Follow the conventions in `AGENTS.md`. The public API is what `microcore/__init__.py` exports,
  and the code must keep working on Python 3.10.
- Keep the change focused on the issue. Don't refactor or reformat code you don't need to touch.
- `requirements/tests.txt` is installed, plus `flake8` and `black`. Your shell is limited to
  `python`, `pytest`, `flake8`, `black`, `pylint` and read-only `git` (`diff`, `status`, `log`,
  `show`); use Read, Glob and Grep to explore. You have no web access: if the task needs
  information that is in neither the issue nor the repository, stop and report it as below.
- Before finishing, run `pytest` (the basic suite) and `flake8 microcore`, and fix any failures
  your change caused. Add or update tests when the change alters behavior.
- Review your own diff with `git diff`.

## Output

- When the change is done, write the pull request description to `pr-body.md` in the repository
  root (the workflow removes it before committing):

  ```markdown
  ## Changes
  - what changed, one line per logical change

  ## Testing
  - the commands you ran and their results
  ```

  Plain Markdown, no attribution lines or "generated with" footers, and no "Closes #N" line, which
  the workflow adds.
- If you can't implement the issue (it's unclear, contradicts itself, needs information you don't
  have, or needs changes to `.github/`, `.claude/` or `.mcp.json`), write `blocked.md` instead,
  explaining what's missing. The workflow then discards your changes and posts it on the issue.
