"""
Class imbalance handling — applied ONLY to the training split.

Reads the already-split, already-scaled train.csv (produced by split.py)
and applies SMOTE selectively: only to classes with enough real samples
to interpolate meaningfully. Tiny classes are left as-is and handled via
class_weight at training time instead (Phase 4).

CRITICAL: this script must run AFTER split.py, and must only ever touch
train.csv. val.csv and test.csv are never resampled — they must reflect
the TRUE, real-world class distribution, because that's what they're for:
measuring how the model performs on realistic, unaltered data.

Usage:
    python src/data/handle_imbalance.py
"""

import json
import pandas as pd
from imblearn.over_sampling import SMOTE

TRAIN_IN_PATH = "data/processed/train.csv"
TRAIN_OUT_PATH = "data/processed/train_resampled.csv"
MULTICLASS_MAP_PATH = "models/metadata/label_map_multiclass.json"

# Classes below this many real training samples are NOT touched by SMOTE —
# there aren't enough real points to interpolate meaningfully between.
# They're left as-is and handled by class_weight during model training
# instead (Phase 4), rather than manufacturing synthetic noise here.
MIN_SAMPLES_FOR_SMOTE = 200

# Don't fully balance to 1:1 — cap the oversampling target at this
# fraction of the majority class size, so we're not manufacturing an
# absurd volume of synthetic data for classes that are naturally rare.
MAX_TARGET_FRACTION_OF_MAJORITY = 0.10


def main():
    print(f"Loading {TRAIN_IN_PATH}...")
    train_df = pd.read_csv(TRAIN_IN_PATH, low_memory=False)
    print(f"Loaded shape: {train_df.shape}")

    with open(MULTICLASS_MAP_PATH) as f:
        multiclass_map = json.load(f)
    inv_map = {v: k for k, v in multiclass_map.items()}

    # -------------------------------------------------------------
    # Step 1 — inspect the current TRAIN-ONLY class distribution
    # -------------------------------------------------------------
    # We compute this from train.csv, which was already split away from
    # val/test by split.py. Everything from here on operates exclusively
    # on this training data — val.csv and test.csv are never opened by
    # this script at all, which is the structural guarantee against
    # accidentally resampling data that must stay untouched.
    counts = train_df["label_multiclass"].value_counts()
    print("\nTrain split class distribution before resampling:")
    for cls_id, count in counts.items():
        print(f"  {inv_map[cls_id]}: {count:,}")

    majority_count = counts.max()
    target_count = int(majority_count * MAX_TARGET_FRACTION_OF_MAJORITY)

    # -------------------------------------------------------------
    # Step 2 — decide which classes get SMOTE, which stay untouched
    # -------------------------------------------------------------
    # A class is only eligible for SMOTE if it has enough real samples
    # to interpolate meaningfully AND is currently below our target —
    # oversampling a class that's already above the target would be
    # pointless (or would need UNDER-sampling instead, which SMOTE
    # doesn't do).
    sampling_strategy = {}
    for cls_id, count in counts.items():
        if count >= MIN_SAMPLES_FOR_SMOTE and count < target_count:
            sampling_strategy[cls_id] = target_count

    skipped_too_small = [inv_map[c] for c, n in counts.items() if n < MIN_SAMPLES_FOR_SMOTE]
    print(f"\nClasses SMOTE will oversample: {[inv_map[c] for c in sampling_strategy]}")
    print(f"Classes left untouched (too few real samples, {MIN_SAMPLES_FOR_SMOTE} minimum): {skipped_too_small}")
    print("These will instead be handled via class_weight during model training (Phase 4).")

    if not sampling_strategy:
        print("\nNo classes qualify for SMOTE with current thresholds — saving train split unchanged.")
        train_df.to_csv(TRAIN_OUT_PATH, index=False)
        return

    # -------------------------------------------------------------
    # Step 3 — apply SMOTE to the eligible classes only
    # -------------------------------------------------------------
    # Separate features (X) from the multiclass target (y) — SMOTE
    # operates on the feature space to find nearest neighbors and
    # interpolate between them, so it needs pure numeric feature columns.
    # `label_binary` is dropped here temporarily since it's derived
    # entirely FROM label_multiclass, and we regenerate it after
    # resampling rather than resampling it independently (which could
    # otherwise let the two targets drift out of sync).
    feature_cols = [c for c in train_df.columns if c not in ("label_binary", "label_multiclass")]
    X = train_df[feature_cols]
    y = train_df["label_multiclass"]

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # -------------------------------------------------------------
    # Step 4 — reassemble, regenerate label_binary, and save
    # -------------------------------------------------------------
    resampled_df = pd.DataFrame(X_resampled, columns=feature_cols)
    resampled_df["label_multiclass"] = y_resampled
    # Regenerate label_binary from the (possibly new, synthetic) multiclass
    # labels rather than resampling label_binary separately — this
    # guarantees the two targets can never disagree with each other.
    benign_class_id = multiclass_map.get("Benign")
    resampled_df["label_binary"] = (resampled_df["label_multiclass"] != benign_class_id).astype(int)

    print(f"\nShape before resampling: {train_df.shape}")
    print(f"Shape after resampling: {resampled_df.shape}")

    print("\nTrain split class distribution AFTER resampling:")
    new_counts = resampled_df["label_multiclass"].value_counts()
    for cls_id, count in new_counts.items():
        print(f"  {inv_map[cls_id]}: {count:,}")

    resampled_df.to_csv(TRAIN_OUT_PATH, index=False)
    print(f"\nSaved resampled training set to {TRAIN_OUT_PATH}")
    print("val.csv and test.csv were never touched — they retain the true, real-world class distribution.")


if __name__ == "__main__":
    main()