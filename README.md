# bams-ai-data-exploration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NCAR/bams-ai-data-exploration/blob/main/notebooks/02-generate-embeddings/generate_dinov3_embeddings.ipynb)
[![Tests](https://github.com/NCAR/bams-ai-data-exploration/actions/workflows/tests.yml/badge.svg)](https://github.com/NCAR/bams-ai-data-exploration/actions/workflows/tests.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Getting Started

These steps are intended for someone running the project locally for the first time.

### 1) Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### 2) Clone the repository

```bash
git clone https://github.com/NCAR/bams-ai-data-exploration.git
cd bams-ai-data-exploration
```

### 3) Install dependencies

```bash
uv sync
```

This creates or updates a local virtual environment and installs dependencies from `pyproject.toml` and `uv.lock`.

### 4) Run notebooks

Start Jupyter and open any notebook under the `notebooks/` directory.

```bash
uv run jupyter notebook
```

### 5) Run the Python entry point (optional)

```bash
uv run python main.py
```

## What To Do Next

After your environment is working, a good next path is:

1. Open `notebooks/01-prepare-data/` and run the notebooks from top to bottom.
2. Move to `notebooks/02-generate-embeddings/`.
3. Use the Colab badge above if you prefer a hosted environment instead of local setup.

## How to Review and Contribute

### Review local changes

```bash
git status
git diff
```

### Run tests before opening a PR

```bash
uv run pytest tests/ -v
```

### Create a contribution branch

```bash
git checkout -b <short-feature-name>
```

### Commit and push your changes

```bash
git add .
git commit -m "Describe your change"
git push -u origin <short-feature-name>
```

### Open a pull request

1. Open your branch in GitHub.
2. Create a PR into `main`.
3. Include:
   - What changed
   - Why it changed
   - How you tested it

## Development Tips

- Keep dependencies updated with `uv lock` and `uv sync`.
- Keep notebook and script changes focused so they are easier to review.
