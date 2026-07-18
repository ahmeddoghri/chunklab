# Contributing

Thanks for taking a look. New chunking strategies are especially welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Before opening a pull request

- Keep changes focused. One logical change per PR, not a drive-by rewrite.
- Add or update tests for any behaviour you change. CI runs `pytest` on
  Python 3.9, 3.11, and 3.13, plus the example and benchmark, so it will
  find you.
- Run `ruff check .` and `pytest -q` locally before you push.
- A new chunker must return chunks whose `text` matches `DOCUMENT[start:end]`
  exactly (the tests enforce this) so the answer-coverage metric stays honest.
  Add it to the benchmark so its coverage shows up next to the others.

## Reporting bugs

Open an issue with a minimal reproduction, the expected versus actual result,
and your Python version. For security issues see [SECURITY.md](SECURITY.md).
