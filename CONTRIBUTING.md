# Contributing to extracture

Thanks for your interest in contributing!

## How to Contribute

1. **Fork** the repository
2. **Create a branch** for your feature/fix: `git checkout -b my-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest tests/ -v`
5. **Run linting**: `ruff check src/ tests/`
6. **Commit** with a clear message
7. **Push** to your fork and open a **Pull Request**

## Pull Request Rules

- All PRs require **approval from a maintainer** before merging
- All CI checks (tests, lint, type check) must pass
- PRs should include tests for new functionality
- Keep PRs focused — one feature or fix per PR

## Development Setup

```bash
git clone https://github.com/extracture/extracture.git
cd extracture
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

- We use `ruff` for linting
- We use `mypy` for type checking
- Follow existing code patterns
- Add type hints to all public functions

## Reporting Issues

Open an issue on GitHub with:
- What you expected
- What happened instead
- Steps to reproduce
- Python version and OS
