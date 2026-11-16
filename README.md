# Spam classifier

This repository contains the initial data-loading and baseline checkpoint for the CECS 456 final project proposal.

## Run the checkpoint

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python analyze_spambase.py
```

The analysis uses the included UCI Spambase source files in `data/`. The dataset contains 4,601 email rows, 57 numeric predictors, and a binary `spam` target (`1` = spam, `0` = not spam). See [`reports/checkpoint.md`](reports/checkpoint.md) for the recorded results and interpretation.

## Data source

UCI Machine Learning Repository, [Spambase](https://archive.ics.uci.edu/dataset/94/spambase). The source data and attribute documentation are included under `data/` for reproducibility.
