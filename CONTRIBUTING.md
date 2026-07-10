# Contributing to JAFF

Thanks for your interest in contributing! This is a quick-start summary — the
full guides live in [`docs/development/`](docs/development/) and online in the
[Documentation](https://jaff-chemistry.github.io/jaff/).

## Ways to Contribute

- **Report bugs** — open an issue with steps to reproduce.
- **Suggest features** — open an issue to discuss before building anything large.
- **Fix issues / add features** — pick an issue, then send a PR.
- **Improve docs** — corrections and clarifications are always welcome.

For major changes, open an issue first so the direction can be agreed on before
you invest time.

## Development Setup

Fork the repo, clone your fork, and install in editable mode with dev tooling:

```bash
git clone https://github.com/YOUR_USERNAME/jaff.git
cd jaff
pip install -e ".[dev]"
```

Requires **Python 3.11+**. Full instructions (venv, uv, conda, IDE setup) are in
the [Installation Guide](docs/development/installation.md).

## Workflow

1. **Branch** off an up-to-date `main` — never commit to `main` directly. Prefix
   the branch with its purpose: `feature/`, `bug-fix/`, `docs/`, `refactor/`,
   `test/`, or `chore/`.

   ```bash
   git checkout main && git pull upstream main
   git checkout -b feature/short-description
   ```

2. **Commit** with clear messages explaining _what_ and _why_. Use Conventional
   Commit types (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`):

   ```bash
   git commit -m "feat: add support for GPU code generation"
   ```

3. **Check** that tests pass and code is formatted before opening a PR:

   ```bash
   pytest          # run the test suite
   ruff check .    # lint
   ruff format .   # format
   ```

4. **Open a PR** against `jaff-chemistry/jaff:main`. Summarize the change,
   reference related issues (`Fixes #123`), and note anything reviewers should
   look at. All CI checks (tests on Linux/macOS/Windows × Python 3.11–3.13, docs
   build, notebook execution) must pass before merge.

## Standards

- **Code style** — Ruff (90-char lines, double quotes), built-in generics and
  `X | None` unions, NumPy-style docstrings. See the
  [Code Style Guide](docs/development/code-style.md).
- **Tests** — pytest, one behaviour per test, descriptive names. New code should
  be fully covered. See the [Testing Guide](docs/development/testing.md).
- **Codebase layout** — see the
  [Codebase Structure](docs/development/codebase-structure.md) guide.

## Getting Help

- **GitHub Issues** — bug reports and feature requests.
- **GitHub Discussions** — questions about the codebase or usage.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
