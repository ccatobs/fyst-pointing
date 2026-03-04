# fyst-pointing

Telescope pointing utilities for the FYST/CCAT submillimeter telescope.

## Installation

```bash
pip install "fyst-pointing @ git+https://github.com/ccatobs/fyst-pointing.git"
```

For usage documentation, see [fyst-pointing.readthedocs.io](https://fyst-pointing.readthedocs.io/).

## Development

```bash
git clone https://github.com/ccatobs/fyst-pointing.git
cd fyst-pointing
pip install -e ".[dev]"

pytest tests/
ruff check . && ruff format --check .
```

### Cross-validation tests

Cross-validation tests verify numerical correctness against independent implementations.
They are gated behind the `--run-slow` flag:

```bash
pytest tests/ --run-slow
```

- **Skyfield**: verifies coordinate transforms against an independent astronomy library
- **scan_patterns**: verifies scan pattern geometry against the legacy CCAT implementation
  (requires `pip install mapping @ git+https://github.com/ccatobs/scan_patterns.git`).
  On Windows or strict-locale environments, you may need to first patch
  `scan_patterns/setup.py` to add `encoding='utf-8'` to the `open()` call
  (Python 3.10+ no longer defaults to UTF-8).
- **KOSMA**: verifies focal plane offset model against the KOSMA telescope control system
