# ROCK Clustering Algorithm

Implementation of the ROCK: A Robust Clustering Algorithm for Categorical Attributes.
Based on the: [ROCK: A Robust Clustering Algorithm for Categorical Attributes - Data Engineering, 1999. Proceedings.](https://theory.stanford.edu/~sudipto/mypapers/categorical.pdf)

## Usage

### Install project

From the root directory:

```bash
> pip install -r requirements.txt

> export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/rock
```

### Run tests

From the root directory:

```bash
> pytest tests
```

### Run experiments

From the root directory:

```bash
> python runner.py --dataset congressional --theta 0.6 --k 2 --approx_fn rational_sub --split_train 0.35
```
