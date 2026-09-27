"""
Data loading module for CyberShield AI.

Discovers CSV(s) under a given directory, loads them with memory-efficient
dtypes, strips whitespace from column names, concatenates them, and reports
per-file row counts plus any schema mismatches between files.

Works whether you have one consolidated file (your current setup) or many
day-wise files (the original CIC-IDS2017 raw release) — it doesn't assume
either shape.

Usage:
    python src/data/load.py                  # loads full data/processed/
    python src/data/load.py --sample         # loads a 10% random sample
    python src/data/load.py --dir data/raw   # point at a different folder
"""

import argparse
import glob
import os
import pandas as pd
import numpy as np


def discover_csv_files(directory: str) -> list[str]:
    """Find every .csv file under `directory`, sorted for reproducible order."""
    pattern = os.path.join(directory, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {directory}")
    return files


def infer_efficient_dtypes(sample_df: pd.DataFrame) -> dict:
    """
    Build a dtype map that downcasts numeric columns to the smallest safe
    type and converts low-cardinality string columns (like the Label
    column) to `category`.

    We infer this from a SAMPLE (not the full file) because reading the
    full file just to decide dtypes would defeat the purpose — the sample
    read itself already uses pandas' default dtypes, which is fine since
    we only use it to inspect ranges and cardinality, not to keep in memory.
    """
    dtype_map = {}
    for col in sample_df.columns:
        col_data = sample_df[col]

        if pd.api.types.is_float_dtype(col_data):
            # float64 -> float32 halves memory for every numeric column.
            # CIC-IDS flow features (durations, byte counts, rates) never
            # need float64 precision — float32 gives ~7 significant digits,
            # far more than these measurements are ever actually accurate to.
            dtype_map[col] = "float32"

        elif pd.api.types.is_integer_dtype(col_data):
            # Downcast integers to the smallest type that fits the observed
            # range. Most flow-count columns (packet counts, flag counts)
            # fit comfortably in int32, many even in int16.
            col_min, col_max = col_data.min(), col_data.max()
            if col_min >= 0:
                if col_max < 65535:
                    dtype_map[col] = "uint16"
                else:
                    dtype_map[col] = "uint32"
            else:
                dtype_map[col] = "int32"

        elif pd.api.types.is_object_dtype(col_data):
            # The Label column (and any other string columns) is low-
            # cardinality — CIC-IDS2017 has ~15 distinct label values across
            # millions of rows. `category` stores each unique string once
            # and represents every row as a small integer code instead of
            # a full Python string object, which is dramatically cheaper.
            n_unique = col_data.nunique()
            if n_unique < 100:
                dtype_map[col] = "category"
            # else: leave as default object dtype (rare for this dataset)

    return dtype_map


def load_all_csvs(directory: str, sample_frac: float = None, seed: int = 42) -> pd.DataFrame:
    files = discover_csv_files(directory)
    print(f"Discovered {len(files)} CSV file(s) in {directory}:")
    for f in files:
        print(f"  - {f}")

    # Read a small sample of the FIRST file to infer dtypes and reference
    # schema, before committing to loading everything.
    reference_sample = pd.read_csv(files[0], nrows=5000)
    reference_sample.columns = reference_sample.columns.str.strip()
    reference_columns = set(reference_sample.columns)
    dtype_map = infer_efficient_dtypes(reference_sample)

    frames = []
    total_rows = 0

    for f in files:
        # Strip whitespace from column names via `skipinitialspace` at read
        # time, then again explicitly after — CIC-IDS2017 is notorious for
        # inconsistent leading spaces in column headers across day-files.
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()

        # Schema mismatch check — compare this file's columns against the
        # reference file's columns. This catches the case where one day's
        # CSV has an extra/missing/renamed column, which would otherwise
        # silently produce NaN-filled columns after concatenation.
        this_columns = set(df.columns)
        missing = reference_columns - this_columns
        extra = this_columns - reference_columns
        if missing or extra:
            print(f"  SCHEMA MISMATCH in {f}:")
            if missing:
                print(f"    Missing columns vs reference: {missing}")
            if extra:
                print(f"    Extra columns vs reference: {extra}")
        else:
            print(f"  {f}: {len(df):,} rows — schema OK")

        total_rows += len(df)

        # Apply the dtype map only to columns that exist in this file and
        # are present in our inferred map, so a mismatched file doesn't
        # crash the whole load.
        applicable_dtypes = {c: t for c, t in dtype_map.items() if c in df.columns}
        try:
            df = df.astype(applicable_dtypes)
        except (ValueError, TypeError) as e:
            # If downcasting fails on this file (e.g. an unexpected value
            # outside the inferred integer range), fall back to leaving
            # that file's dtypes as pandas' defaults rather than crashing
            # the whole pipeline.
            print(f"    Warning: dtype downcast failed for {f} ({e}); keeping default dtypes")

        frames.append(df)

    print(f"\nTotal rows across all files: {total_rows:,}")

    combined = pd.concat(frames, ignore_index=True)

    if sample_frac is not None:
        before = len(combined)
        combined = combined.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        print(f"Sampled {sample_frac:.0%}: {before:,} -> {len(combined):,} rows")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Load CIC-IDS2017 data efficiently")
    parser.add_argument("--dir", default="data/processed",
                         help="Directory containing CSV file(s) to load (default: data/processed)")
    parser.add_argument("--sample", action="store_true",
                         help="Load only a 10%% random sample, for fast local development")
    args = parser.parse_args()

    sample_frac = 0.10 if args.sample else None

    df = load_all_csvs(args.dir, sample_frac=sample_frac)

    mem_bytes = df.memory_usage(deep=True).sum()
    print(f"\nFinal shape: {df.shape}")
    print(f"Memory usage after dtype optimization: {mem_bytes / 1e6:.1f} MB")
    print(f"Dtypes:\n{df.dtypes.value_counts()}")


if __name__ == "__main__":
    main()