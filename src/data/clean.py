"""
Cleaning stage of the CyberShield AI preprocessing pipeline.

Operations run in this fixed order, for reasons explained below each one:
  1. Infinite values -> NaN
  2. Missing values (drop rows)
  3. Exact duplicate rows
  4. Constant columns
  5. Near-constant columns (below a variance threshold)
  6. Negative values in features that should be non-negative

Every run   appends a timestamped, numbered decision entry to docs/decisions.md
so the exact counts at each stage are on record for the report — not just
printed to a terminal that gets closed and forgotten.

Usage:
    python src/data/clean.py
    python src/data/clean.py --variance-threshold 0.02
"""

import argparse
import datetime
import os
import numpy as np
import pandas as pd

IN_PATH = "data/processed/cicids2017_clean.csv"   # output of your earlier prepare_dataset.py run
OUT_PATH = "data/processed/cicids2017_stage2_clean.csv"
DECISIONS_LOG = "docs/decisions.md"

# Column-name substrings that identify features which should never be
# negative for a network flow. This is a heuristic keyword match rather
# than a hardcoded exact list, because CIC-IDS2017 column names vary
# slightly in capitalization/spacing across dataset releases.
NON_NEGATIVE_KEYWORDS = [
    "duration", "packets", "bytes", "length", "iat", "count",
    "bulk", "active", "idle", "min", "max", "mean", "std",
]


def get_non_negative_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns whose name suggests they must be >= 0."""
    cols = []
    for c in df.columns:
        c_lower = c.lower()
        if pd.api.types.is_numeric_dtype(df[c]) and any(kw in c_lower for kw in NON_NEGATIVE_KEYWORDS):
            cols.append(c)
    return cols


def clean_data(df: pd.DataFrame, variance_threshold: float = 0.01) -> tuple[pd.DataFrame, dict]:
    report = {"initial_shape": df.shape}

    # -----------------------------------------------------------------
    # Step 1 — Infinite values -> NaN
    # -----------------------------------------------------------------
    # CICFlowMeter computes rate-based features (e.g. Flow Bytes/s) by
    # dividing byte/packet counts by flow duration. A flow lasting 0
    # microseconds produces a division by zero, which pandas/numpy read
    # in from CSV as the literal string "Infinity", parsed as np.inf.
    # We convert these to NaN FIRST so that step 2 can catch them using
    # a single, uniform "is this row broken" check — if we skipped this
    # step, dropna() would silently leave inf values in the data, since
    # inf is not NaN and dropna() only targets NaN by default.
    n_inf_before = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    df = df.replace([np.inf, -np.inf], np.nan)
    report["infinite_values_converted"] = int(n_inf_before)
    print(f"Step 1 — Converted {n_inf_before:,} infinite values to NaN")

    # -----------------------------------------------------------------
    # Step 2 — Missing values (drop rows)
    # -----------------------------------------------------------------
    # Runs SECOND, immediately after step 1, because inf values are now
    # NaN and will be caught by the same dropna() call as genuinely
    # missing values — one pass handles both problems. We drop rather
    # than impute here because flow-level network features (durations,
    # byte counts) don't have a sensible "typical" value to impute with
    # that wouldn't distort the attack-detection signal; a fabricated
    # duration could accidentally look like a real flow pattern.
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    report["rows_dropped_missing"] = rows_dropped
    print(f"Step 2 — Dropped {rows_dropped:,} rows with missing/inf values ({rows_dropped/rows_before:.2%})")

    # -----------------------------------------------------------------
    # Step 3 — Exact duplicate rows
    # -----------------------------------------------------------------
    # Runs THIRD, after rows are cleaned of NaN/inf — not before. Two
    # rows that differ only in a since-removed NaN pattern could look
    # like duplicates prematurely, or fail to match as duplicates when
    # they should, if compared before this cleanup. Deduplicating on the
    # now-consistent data gives a correct duplicate count.
    rows_before = len(df)
    df = df.drop_duplicates()
    dupes_dropped = rows_before - len(df)
    report["duplicate_rows_dropped"] = dupes_dropped
    print(f"Step 3 — Dropped {dupes_dropped:,} exact duplicate rows")

    # -----------------------------------------------------------------
    # Step 4 — Constant columns
    # -----------------------------------------------------------------
    # Runs FOURTH — only after rows are finalized, because a column's
    # constant-ness must be evaluated on the data you're actually
    # keeping. A column could look constant before row cleaning simply
    # because all its varying values lived in rows that got dropped in
    # steps 1-3; checking now avoids that false conclusion.
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=constant_cols)
    report["constant_columns_dropped"] = constant_cols
    print(f"Step 4 — Dropped {len(constant_cols)} constant columns: {constant_cols}")

    # -----------------------------------------------------------------
    # Step 5 — Near-constant columns (variance threshold)
    # -----------------------------------------------------------------
    # Runs FIFTH, after exact-constant columns are already removed —
    # this avoids redundant variance computation on columns we already
    # know are worthless. A near-constant column (e.g. 99.9% zeros with
    # a handful of outliers) carries almost no discriminative signal for
    # the model but still costs training time and can destabilize
    # distance-based or gradient-based learners. We normalize by dividing
    # variance by the mean of squared values (a coefficient-of-variation-
    # style measure) rather than using raw variance, because raw variance
    # is scale-dependent — a byte-count column and a ratio column live on
    # totally different numeric scales, and raw variance would unfairly
    # flag small-scale columns as "near-constant" when they aren't.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    near_constant_cols = []
    for col in numeric_cols:
        col_data = df[col].astype(np.float64)
        mean_sq = (col_data ** 2).mean()
        if mean_sq == 0:
            continue  # already caught by step 4 if truly constant at zero
        normalized_variance = col_data.var() / mean_sq
        if normalized_variance < variance_threshold:
            near_constant_cols.append(col)
    df = df.drop(columns=near_constant_cols)
    report["near_constant_columns_dropped"] = near_constant_cols
    print(f"Step 5 — Dropped {len(near_constant_cols)} near-constant columns "
          f"(threshold={variance_threshold}): {near_constant_cols}")

    # -----------------------------------------------------------------
    # Step 6 — Negative values in features that should be non-negative
    # -----------------------------------------------------------------
    # Runs LAST, deliberately — it operates on the final column set from
    # steps 4-5, so we never waste effort correcting a column that was
    # about to be dropped anyway. Negative values here are a known
    # CICFlowMeter quirk (rare) rather than genuine signal — a negative
    # "Flow Duration" or "Packet Length" is physically meaningless. We
    # clip to zero rather than drop the row, because a single spurious
    # negative field on an otherwise-valid flow shouldn't cost us a
    # whole row of legitimate data.
    non_negative_cols = get_non_negative_columns(df)
    negative_counts = {}
    for col in non_negative_cols:
        n_negative = (df[col] < 0).sum()
        if n_negative > 0:
            negative_counts[col] = int(n_negative)
            df[col] = df[col].clip(lower=0)
    report["negative_values_clipped"] = negative_counts
    total_negatives = sum(negative_counts.values())
    print(f"Step 6 — Clipped {total_negatives:,} negative values across "
          f"{len(negative_counts)} columns: {negative_counts}")

    report["final_shape"] = df.shape
    return df, report


def log_decision(report: dict, variance_threshold: float):
    """Append a timestamped, numbered entry to docs/decisions.md."""
    os.makedirs(os.path.dirname(DECISIONS_LOG), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    entry = f"""
## Data Cleaning Run — {timestamp}

- Initial shape: {report['initial_shape']}
- Infinite values converted to NaN: {report['infinite_values_converted']:,}
- Rows dropped (missing/inf): {report['rows_dropped_missing']:,}
- Duplicate rows dropped: {report['duplicate_rows_dropped']:,}
- Constant columns dropped ({len(report['constant_columns_dropped'])}): {report['constant_columns_dropped']}
- Near-constant columns dropped (threshold={variance_threshold}, {len(report['near_constant_columns_dropped'])}): {report['near_constant_columns_dropped']}
- Negative values clipped to zero: {report['negative_values_clipped']}
- Final shape: {report['final_shape']}
"""
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"\nDecision log appended to {DECISIONS_LOG}")


def main():
    parser = argparse.ArgumentParser(description="Clean CIC-IDS2017 dataset")
    parser.add_argument("--variance-threshold", type=float, default=0.01,
                         help="Normalized variance threshold below which a column is dropped as near-constant")
    args = parser.parse_args()

    print(f"Loading {IN_PATH}...")
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Loaded shape: {df.shape}\n")

    df_clean, report = clean_data(df, variance_threshold=args.variance_threshold)

    df_clean.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUT_PATH} — final shape {df_clean.shape}")

    log_decision(report, args.variance_threshold)


if __name__ == "__main__":
    main()