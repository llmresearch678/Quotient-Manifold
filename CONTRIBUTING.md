# Contributing to Quotient Diffusion Models

Thank you for your interest in contributing to QDM! This document provides guidelines
for contributing code, experiments, and documentation.

## Development Setup

```bash
git clone https://github.com/llmresearch678/Quotient-Manifold.git
cd Quotient-Manifold
pip install -e ".[dev]"
pre-commit install
```

## Code Style

We use `black` (line length 100) and `isort`:

```bash
black qdm/ tests/ scripts/
isort qdm/ tests/ scripts/
flake8 qdm/ tests/ scripts/
```

## Running Tests

```bash
pytest tests/ -v --cov=qdm --cov-report=term-missing
```

All tests must pass before submitting a pull request.

## Adding a New Symmetry Group

1. Add the generator in `qdm/geometry/lie_groups.py`:
   ```python
   class MySE2Generator(LieGroupGenerator):
       def __call__(self, x: torch.Tensor) -> torch.Tensor:
           # Return V_x: R^{d×k} generator matrix
           ...
   ```

2. Register it in `qdm/geometry/__init__.py`.

3. Add tests in `tests/test_qdm.py::TestMySE2Generator`.

## Reporting Issues

Please open a GitHub Issue with:
- Python version and OS
- PyTorch version
- Minimal reproducible example
- Full traceback

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Code is formatted (`black`, `isort`)
- [ ] New functionality has tests
- [ ] Docstrings are updated
- [ ] CHANGELOG.md entry added (if applicable)
