# etomofiles

[![License](https://img.shields.io/pypi/l/etomofiles.svg?color=green)](https://github.com/davidetorre99/etomofiles/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/etomofiles.svg?color=green)](https://pypi.org/project/etomofiles)
[![Python Version](https://img.shields.io/pypi/pyversions/etomofiles.svg?color=green)](https://python.org)
[![CI](https://github.com/davidetorre99/etomofiles/actions/workflows/ci.yml/badge.svg)](https://github.com/davidetorre99/etomofiles/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/davidetorre99/etomofiles/branch/main/graph/badge.svg)](https://codecov.io/gh/davidetorre99/etomofiles)

A Python package for reading IMOD etomo alignment files into pandas DataFrames 

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork davidetorre99/etomofiles --clone
# or just
# gh repo clone davidetorre99/etomofiles
cd etomofiles
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
