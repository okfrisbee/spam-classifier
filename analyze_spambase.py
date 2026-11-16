"""Load UCI Spambase and evaluate the proposed logistic-regression baseline."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "spambase.data"
NAMES_PATH = ROOT / "data" / "spambase.names"


def load_data() -> pd.DataFrame:
    names_text = NAMES_PATH.read_text()
    feature_names = re.findall(r"^\s*(.+):\s+continuous\.$", names_text, flags=re.MULTILINE)
    if len(feature_names) != 57:
        raise ValueError(f"Expected 57 feature names in {NAMES_PATH}, found {len(feature_names)}")
    columns = [name.strip() for name in feature_names] + ["spam"]
    data = pd.read_csv(DATA_PATH, header=None, names=columns, skipinitialspace=True)
    data["spam"] = data["spam"].astype("int64")
    return data


def main() -> None:
    data = load_data()
    X = data.drop(columns="spam")
    y = data["spam"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=42))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"dataset_shape={data.shape}")
    print(f"feature_count={X.shape[1]}")
    print(f"dataframe_dtypes={data.dtypes.value_counts().to_dict()}")
    print(f"missing_cells={int(data.isna().sum().sum())}")
    print(f"missing_by_column={int(data.isna().sum().gt(0).sum())}")
    print(f"class_counts={y.value_counts().sort_index().to_dict()}")
    print(f"class_percentages={(y.value_counts(normalize=True).sort_index() * 100).round(2).to_dict()}")
    print(f"train_rows={len(X_train)} train_class_counts={y_train.value_counts().sort_index().to_dict()}")
    print(f"test_rows={len(X_test)} test_class_counts={y_test.value_counts().sort_index().to_dict()}")
    print(f"baseline_precision_spam_positive={precision_score(y_test, predictions, pos_label=1):.4f}")
    print(f"baseline_f1={f1_score(y_test, predictions):.4f}")
    print(f"baseline_roc_auc={roc_auc_score(y_test, probabilities):.4f}")
    print(f"baseline_accuracy={accuracy_score(y_test, predictions):.4f}")
    print(f"baseline_confusion_matrix_labels_0_1={confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()}")


if __name__ == "__main__":
    main()
