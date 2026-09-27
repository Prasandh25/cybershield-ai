"""
Splitting and scaling stage of the CyberShield AI preprocessing pipeline.

Produces:
  - data/processed/train.csv, val.csv, test.csv  (70/15/15, stratified)
  - models/artifacts/scaler.joblib               (fitted on TRAIN ONLY)
  - models/metadata/feature_columns.json         (exact column order)

Usage:
    python src/data/split.py
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IN_PATH = "data/processed/cicids2017_labeled.csv"
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
SCALER_PATH = "models/artifacts/scaler.joblib"
FEATURE_COLUMNS_PATH = "models/metadata/feature_columns.json"

RANDOM_SEED = 42


def main():
    print(f"Loading {IN_PATH}...")
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Loaded shape: {df.shape}\n")

    # -------------------------------------------------------------
    # Step 1 — identify feature columns (everything except labels)
    # -------------------------------------------------------------
    # Detect the raw label column by name the same way labels.py did —
    # hardcoding "Label" here caused a real bug: this dataset's raw
    # label column has a slightly different name/casing, so it silently
    # survived a hardcoded exclusion list and reached StandardScaler as
    # a string column. Detecting it dynamically, and ALSO excluding any
    # remaining non-numeric column as a safety net, closes both the
    # specific bug and the general class of bug it represents.
    raw_label_col = "Label" if "Label" in df.columns else [
        c for c in df.columns if c not in ("label_binary", "label_multiclass")
        and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if isinstance(raw_label_col, list):
        raw_label_col = raw_label_col[0] if raw_label_col else None

    non_feature_columns = {raw_label_col, "label_binary", "label_multiclass"}
    non_feature_columns.discard(None)

    # Safety net: explicitly drop ANY remaining non-numeric column from
    # the feature set, whatever it's named, so a future dataset variant
    # with yet another oddly-named text column can't reintroduce this
    # exact class of bug silently.
    non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                         and c not in non_feature_columns]
    if non_numeric_cols:
        print(f"Excluding additional non-numeric columns found: {non_numeric_cols}")
        non_feature_columns.update(non_numeric_cols)

    feature_columns = [c for c in df.columns if c not in non_feature_columns]
    print(f"Raw label column detected as: {raw_label_col!r}")
    print(f"Identified {len(feature_columns)} feature columns")

    X = df[feature_columns]
    y_binary = df["label_binary"]
    y_multiclass = df["label_multiclass"]

    # -------------------------------------------------------------
    # Step 2 — SPLIT FIRST, before any scaling happens
    # -------------------------------------------------------------
    # We split into train (70%) and a temporary 30% chunk first, then
    # split that 30% chunk in half to get val (15%) and test (15%).
    # `stratify=y_multiclass` ensures each split preserves the original
    # class proportions — critical here, since some attack classes
    # (Heartbleed: 11 rows total) are so rare that a non-stratified
    # random split could easily place ALL of them in one split and
    # zero in another, making that class untrainable or unevaluatable.
    X_train, X_temp, y_bin_train, y_bin_temp, y_multi_train, y_multi_temp = train_test_split(
        X, y_binary, y_multiclass,
        test_size=0.30,
        stratify=y_multiclass,
        random_state=RANDOM_SEED,
    )
    X_val, X_test, y_bin_val, y_bin_test, y_multi_val, y_multi_test = train_test_split(
        X_temp, y_bin_temp, y_multi_temp,
        test_size=0.50,  # half of the remaining 30% = 15% of the original total
        stratify=y_multi_temp,
        random_state=RANDOM_SEED,
    )
    print(f"Train: {X_train.shape[0]:,} rows | Val: {X_val.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    # -------------------------------------------------------------
    # Step 3 — fit the scaler on TRAIN ONLY, then apply to all three
    # -------------------------------------------------------------
    # This is the step order that matters most in this whole script.
    # `.fit()` computes each feature's mean and standard deviation FROM
    # WHATEVER DATA YOU GIVE IT. Calling `.fit()` on X_train computes
    # those statistics using only the 70% of data the model will
    # actually learn from — exactly mirroring the real-world situation
    # where, at deployment time, you have no access whatsoever to
    # future/unseen traffic when you built your preprocessing.
    #
    # `.transform()` (not `.fit_transform()`) is then applied to train,
    # val, AND test — reusing the SAME mean/std computed from train.
    # This is the correct behavior: val and test are meant to simulate
    # "data the model has never seen," so their scaling must use
    # statistics the model already had access to, not their own.
    scaler = StandardScaler()
    scaler.fit(X_train)  # <-- statistics computed from train rows ONLY

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)

    # -------------------------------------------------------------
    # Step 4 — reassemble each split with its labels and save
    # -------------------------------------------------------------
    train_df = X_train_scaled.copy()
    train_df["label_binary"] = y_bin_train
    train_df["label_multiclass"] = y_multi_train

    val_df = X_val_scaled.copy()
    val_df["label_binary"] = y_bin_val
    val_df["label_multiclass"] = y_multi_val

    test_df = X_test_scaled.copy()
    test_df["label_binary"] = y_bin_test
    test_df["label_multiclass"] = y_multi_test

    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"\nSaved train/val/test CSVs to {os.path.dirname(TRAIN_PATH)}/")

    # -------------------------------------------------------------
    # Step 5 — persist the fitted scaler
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved fitted scaler to {SCALER_PATH}")

    # -------------------------------------------------------------
    # Step 6 — persist the exact feature column order
    # -------------------------------------------------------------
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Saved feature column order ({len(feature_columns)} columns) to {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()