# etomofiles

[![License](https://img.shields.io/pypi/l/etomofiles.svg?color=green)](https://github.com/teamtomo/etomofiles/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/etomofiles.svg?color=green)](https://pypi.org/project/etomofiles)
[![Python Version](https://img.shields.io/pypi/pyversions/etomofiles.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/etomofiles/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/etomofiles/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/etomofiles/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/etomofiles)

A Python package for reading IMOD etomo alignment files into pandas DataFrames.

## Overview

`etomofiles` provides a way to extract metadata from etomo alignment directories and convert them into structured pandas DataFrames. 

## Installation

```bash
pip install etomofiles
```

## Quick Start

```python
import etomofiles

# Read etomo alignment data
df = etomofiles.read("/path/to/etomo/directory")

print(df.head())

```

## DataFrame Columns

The resulting DataFrame contains the following columns:

| Column | Description |
|--------|-------------|
| `image_path` | Path to specific image in tilt series (e.g., `TS_001.st[0]`) |
| `idx_tilt` | Index of the tilt image (0-based) |
| `tilt_axis_angle` | Tilt axis rotation angle (degrees) |
| `rawtlt` | Raw tilt angle (degrees) |
| `tlt` | Corrected tilt angle (degrees) |
| `xtilt` | Tilt around x-axis |
| `xf_a11`, `xf_a12`, `xf_a21`, `xf_a22`, `xf_dx`, `xf_dy` | xf transformation matrix elements |
| `excluded` | Boolean indicating if view was excluded |

## xf Utilities

Get xf as numpy array:

```python
import etomofiles

# Read alignment data
df = etomofiles.read("TS_001/")

# Get xf as numpy array:
xf_matrices = etomofiles.df_to_xf(df)

# Each matrix represents an affine transformation:
# [[A11, A12, DX],
#  [A21, A22, DY]]
# where X' = A11*X + A12*Y + DX
#       Y' = A21*X + A22*Y + DY

# Also works directly with files
xf_matrices = etomofiles.df_to_xf("TS_001/TS_001.xf")

# Choose row ordering convention
xf_xy = etomofiles.df_to_xf(df)  # default xy
# Each matrix is [[A11, A12, DX], [A21, A22, DY]]

xf_yx = etomofiles.df_to_xf(df, yx=True)  # yx
# Each matrix is [[A22, A21, DY], [A12, A11, DX]]

```


