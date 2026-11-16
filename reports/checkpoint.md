# Spam classifier project checkpoint

## 1. Data loaded

The UCI Spambase data loads successfully from the included `data/spambase.data` file. The loaded table has **4,601 rows and 58 columns**: 57 predictors plus the `spam` target.

- **Feature types:** 55 continuous numeric features and 2 integer-valued features (`capital_run_length_longest` and `capital_run_length_total`). The target is binary integer (`0` = not spam, `1` = spam). In the loaded pandas table, this is 55 `float64` columns and 3 `int64` columns, including the target.
- **Missing values:** 0 missing cells; no columns contain missing values.
- **Class distribution:** 2,788 not spam (60.6%) and 1,813 spam (39.4%).

## 2. Baseline result

**Spam precision: 0.9209** on a stratified 20% holdout test split (921 test emails; 3,680 train emails), using `random_state=42`. The model is a standardized logistic regression pipeline, with spam (`1`) treated as the positive class and the default 0.5 decision threshold. No threshold or model tuning was performed on the test set.

Additional holdout metrics for context: F1 = 0.9093, ROC-AUC = 0.9702, accuracy = 0.9294. Confusion matrix (true rows and predicted columns, labels `[0, 1]`): `[[530, 28], [37, 326]]`.

## 3. What changed from the proposal

The dataset, target definition, baseline model, and proposed precision metric match the proposal. The UCI ZIP's `spambase.data` begins with a two-value annotation line that is not a feature row. I checked the legacy UCI data endpoint and used its complete 4,601-row file instead; it reproduces the proposal's 2,788/1,813 class counts. The submitted analysis deliberately loads the raw data file directly and does not drop that first real email row.

The proposed model comparisons (tree ensembles, Naive Bayes, and SVM), threshold tuning, and class-weight experiments are not part of this checkpoint yet.

## 4. Blockers

No blockers for the data-loading and baseline checkpoint. The dataset is historical and may not represent current spam, as noted in the proposal; this is a limitation for future interpretation rather than a blocker to reproducing the planned baseline.

## Reproduction

From the repository root, create a virtual environment, install `requirements.txt`, and run `python analyze_spambase.py`. The script prints the shape, pandas dtypes, missing-value counts, class distribution, split counts, and metrics used above.

**Dataset citation:** Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53G6X. Licensed CC BY 4.0.
