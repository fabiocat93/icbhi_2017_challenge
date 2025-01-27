# icbhi_2017_challenge
This repository provides code for pre-processing, training, and testing a model to classify respiratory cycles from the ICBHI 2017 Challenge dataset. It predicts the presence of crackles, wheezes, or both.

## Repo structure
```
icbhi_2017_challenge/
├── src/                    # Source code for the project
│   ├── code/ # Scripts
│   └── test/               # Unit tests
├── docs/                   # Documentation files
├── data/                   # Directory for storing dataset files
├── .github/                # Github actions
├── .gitignore              # Configuration for ignoring files and directories
├── .pre-commit-config.yaml # Configuration for pre-commit hooks
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # Project README file
└── LICENSE                 # License for the project
```

## Scripts
1. **`s00_audio_summary.py`**  
   Generates a "summary" of each audio clip by extracting key statistics and features from the dataset.
2. **`s01_plot_spectrogram.ipynb`**  
   Interactive notebook for visualizing spectrograms to analyze frequency components (useful for initial exploration).
3. **`s02_preprocessing.py`**  
   Handles audio preprocessing tasks such as normalization, filtering, and feature extraction.
4. **`s03_train_dev_test_split.py`**  
   Splits the dataset into training, development, and testing sets based on configurable criteria.
5. **`s04_ml_pipeline.py`**  
   Implements a complete machine learning pipeline, including model training and evaluation. For some notes on this, please check [this notebook](docs/notes.md). 
6. **`s05_inference.py`**  
   Performs inference using trained models on new audio data.


## Setup instructions
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
- Generate and view code documentation:
```bash
poetry run pdoc src/senselab -t docs_style/pdoc-theme --docformat google
```
