# Self-Driving CS361 – Environment Setup

## Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- `conda` available on your `PATH`

## Create and activate the environment

Run this **once** from the repo root:

```bash
conda env create -f environment.yml
```

Then activate it every time you work on the project:

```bash
conda activate selfdriving-cs361
```

## Updating the environment

If new packages are added to `environment.yml`, sync with:

```bash
conda env update -f environment.yml --prune
```
