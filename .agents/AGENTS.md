# SatSim Project Coding Standards (§11)

These rules apply to ALL code written in this workspace.

## Language & Types
- Python 3.11+ only. Full type hints on every public function and method signature.
- Use `from __future__ import annotations` in every module for deferred annotation evaluation.

## Data Transfer Objects
- Use `pydantic.BaseModel` or `dataclasses.dataclass` for all config and data-transfer objects.
- No bare `dict` objects passed between modules. If a function receives or returns structured data, define a model for it.

## Logging
- Use `structlog` (JSON-formatted) exclusively. No `print()` statements in production code.
- Import via `from satsim.logging import get_logger; logger = get_logger(__name__)`.

## Documentation
- Every module must have a module-level docstring explaining its purpose and main public API.
- Every public function and method must have a docstring with `Args:` and `Returns:` sections.
- Private helpers (`_name`) need at minimum a one-line docstring.

## Error Handling
- No silent fallbacks. If a config value is missing or a computation cannot produce a real number, raise an appropriate exception (`ValueError`, `FileNotFoundError`, etc.) — never return `0`, `None`, or `float("nan")` and continue.
- Every `FileNotFoundError` raised by an exporter must include which file is missing and how to produce it.

## Reproducibility
- Every exported dataset file must be accompanied by `config_used.yaml` (the exact config that produced it).
- Running any CLI command twice with the same `--seed` must produce byte-identical outputs.

## Testing
- Every new public function or class ships with at least one test that would fail if replaced with a no-op stub.
- Tests must not depend on external services or the internet.
- Use `tempfile.TemporaryDirectory()` for any test that writes to disk — never write to the repo working tree.

## Dependencies
- All production dependencies must be pinned to exact versions in `pyproject.toml`.
- `gymnasium` is a core dependency (not dev-only), as `envs/routing_env.py` imports it at runtime.
