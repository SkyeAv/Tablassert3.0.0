# Contributing to Tablassert

Thank you for your interest in contributing to Tablassert! This guide covers everything you need to get started.

For full documentation, visit [skyeav.github.io/Tablassert](https://skyeav.github.io/Tablassert/).

## Getting Started

### Prerequisites

- **Python 3.11** or higher
- **[UV](https://docs.astral.sh/uv/)** package manager
- **Git**

### Setup

```bash
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert
uv sync
```

### Optional Extras

All ML, web, and Excel dependencies are included in the core install. The only optional extra is a runtime-compatible Polars build for CPUs without required instructions:

```bash
uv sync --extra rt   # polars[rtcompat]
```

## Development Workflow

### Quick Reference

| Task | Command |
|---|---|
| Run CLI | `uv run tablassert --help` |
| Lint | `uv run ruff check .` |
| Lint (fix) | `uv run ruff check --fix .` |
| Format | `uv run ruff format .` |
| Format check | `uv run ruff format --check .` |
| Type check | `uv run pyright` |
| All checks | `uv run pre-commit run --all-files` |
| Run all tests | `uv run pytest` |
| Run single test | `uv run pytest tests/test_foo.py::test_name` |
| Run by keyword | `uv run pytest -k "test_pattern"` |
| Build | `uv build` |

### Branching

1. Fork the repository
2. Create a branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
3. Make your changes
4. Run all checks before committing:
   ```bash
   uv run ruff check --fix . && uv run ruff format . && uv run pyright && uv run pytest
   ```
5. Push and open a pull request

### Pre-commit Hooks

Pre-commit is configured to run ruff, ruff-format, pyright, and pytest on all Python files. To install the hooks:

```bash
uv run pre-commit install
```

## Pull Requests

- Describe the change and its motivation
- Link any related issues
- Ensure all checks pass (ruff, pyright, pytest)
- Keep PRs focused — one concern per PR is ideal
- If adding a new feature, include tests

## Code Style

### Formatting

Formatting is enforced by **ruff** with these settings:

- Line length: **120**
- Quote style: **double quotes**
- Indent: **4 spaces**
- Target: **Python >=3.11**

### Naming

| Element | Convention | Example |
|---|---|---|
| Functions / variables | `snake_case` | `process_data`, `col_name` |
| Classes | `PascalCase` | `Tcode`, `TablaBase` |
| Module constants | `UPPER_CASE` | `STORE`, `TOKEN_SEP` |

### Comment Markers

Use these prefixes for inline comments:

| Marker | Meaning | Example |
|---|---|---|
| `# ?` | Description or clarification | `# ? strip whitespace from column names` |
| `# !` | Warning or important note | `# ! must run before entity resolution` |
| `# *` | Pipeline stage marker | `# * Stage 2: Entity Resolution` |
| `# TODO:` | Todo item | `# TODO: add fuzzy matching support` |

Do **not** write docstrings on functions. Use a `# ?` comment on the line above instead.

### Type Annotations

**Every variable** must have a type annotation, including locals:

```python
col: str = "name"
df: pl.DataFrame = pl.DataFrame()
result: Optional[int] = None
```

- Use `Optional[T]` and `Union[...]` (not `T | None` or `X | Y`)
- Use `Self` for class methods returning the class type
- Use `Path` (not `str`) for filesystem paths
- Use `# pyright: ignore` to suppress false positives from lazy-loaded modules

### Imports

Every file starts with:

```python
from __future__ import annotations
```

Heavy dependencies are **lazy-loaded** per module:

```python
from typing import TYPE_CHECKING
import lazy_loader as Lazy

if TYPE_CHECKING:
    import polars as pl
else:
    pl = Lazy.load("polars")
```

Lazy-loaded packages: `polars`, `duckdb`, `orjson`, `xxhash`, `polars_hash`, `yaml`, `httpx`, `pyexcel`, `onnxruntime`, `sentence_transformers`

Import order: standard library → blank line → third-party → blank line → local

### Pydantic Models

All models inherit from `TablaBase(BaseModel)`:

```python
from tablassert.models import TablaBase

class MyModel(TablaBase):
    name: str = Field(...)
    description: Optional[str] = Field(None)
```

- Required fields use `Field(...)` (ellipsis sentinel)
- Optional fields use `Optional[T] = Field(None)`
- `extra = "forbid"` — no unknown fields allowed
- `validate_assignment = True` — re-validate on mutation

### Enums

All enums live in `enums.py` and extend `str, Enum`:

```python
class Tokens(str, Enum):
    PIPE = "|"
    COMMA = ","
```

### Error Handling

- Use `RuntimeError` for exceptional cases
- Use `logger.warning()` for non-fatal issues
- Import logger: `from tablassert.log import logger`

## Testing

Tests live in the `tests/` directory at the repo root. Test fixtures are in `tests/fixtures/`.

```bash
# Run all tests
uv run pytest

# Run a specific test
uv run pytest tests/test_lib.py::test_my_function

# Run tests matching a pattern
uv run pytest -k "encoding"

# Run with print output
uv run pytest -s tests/test_lib.py
```

`conftest.py` provides a `fixtures_path` fixture returning `Path(__file__).parent / "fixtures"`.

### Adding Tests

- Place test files in `tests/` following the naming convention `test_<module>.py`
- Use the `fixtures_path` fixture for loading test data
- Add YAML fixture files to `tests/fixtures/` as needed

## AI-Assisted Contributions

Tablassert supports AI-assisted development. The repository includes an `AGENTS.md` file in the root that provides detailed guidance for AI coding tools (GitHub Copilot, Cursor, Claude Code, OpenHands, etc.).

If you use AI tools to contribute:

- Review all generated code before submitting
- Ensure it follows the conventions described above and in `AGENTS.md`
- Run all checks (`ruff`, `pyright`, `pytest`) — AI-generated code often needs style adjustments
- The conventions in this file and `AGENTS.md` help AI tools produce idiomatic Tablassert code

## Reporting Issues

- **Bug reports** and **feature requests**: open an issue at [github.com/SkyeAv/Tablassert/issues](https://github.com/SkyeAv/Tablassert/issues)
- Please include reproduction steps for bugs and a clear description for feature requests

## License

By contributing to Tablassert, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## Code of Conduct

### Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

### Our Standards

Examples of behavior that contributes to a positive environment:

- Demonstrating empathy and kindness toward other people
- Being respectful of differing opinions, viewpoints, and experiences
- Giving and gracefully accepting constructive feedback
- Accepting responsibility and apologizing to those affected by mistakes
- Focusing on what is best not just for us as individuals, but for the overall community

Examples of unacceptable behavior:

- The use of sexualized language or imagery, and sexual attention or advances
- Trolling, insulting or derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the project maintainer at [sgoetz@isbscience.org](mailto:sgoetz@isbscience.org). All complaints will be reviewed and investigated fairly.

Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/), version 2.1.
