# icbhi_2017_challenge
Classification of respiratory sounds using the ICBHI 2017 challenge dataset

## Setup Instructions

### Prerequisites
- Install `Poetry`:
```bash
pip install poetry
```

### Installation
- Clone the repo:
```bash
git clone https://github.com/fabiocat93/icbhi_2017_challenge
```
- Install dependencies (including development and documentation tools):
```bash
cd icbhi_2017_challenge
poetry install --with dev,docs
```

### Optional setup
- Install pre-commit hooks:
```bash
poetry run pre-commit install
```


## Testing and quality checks
### Run Unit Tests
- Run unit tests (in parallel to reduce latency):
```bash
poetry run pytest -n auto
```

### Pre-commit checks
- Run all pre-commit checks (including static type checks with `mypy`, code style checks with `ruff`, and spell checks with `codespell`):
```bash
poetry run pre-commit run --all-files
```

## Documentation
- Generate and view documentation:
```bash
poetry run pdoc src/senselab -t docs_style/pdoc-theme --docformat google
```

## TODO:
- [ ] TBD
