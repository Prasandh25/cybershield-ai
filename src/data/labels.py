"""
Label engineering stage of the CyberShield AI preprocessing pipeline.

Produces TWO target columns from the raw CIC-IDS2017 `Label` column:
  - `label_binary`     : 0 = Benign, 1 = Attack
  - `label_multiclass` : integer code for the attack-taxonomy class
                         (Step 1.2 — PortScan, DoS, DDoS, BruteForce,
                          WebAttack, Infiltration, Bot, Heartbleed, Benign)

Both mappings are saved to models/metadata/ as JSON, so any downstream
code (training, inference, dashboard) reads the SAME mapping rather than
each recomputing its own — a mismatch here would silently corrupt every
prediction's meaning.

Usage:
    python src/data/labels.py
"""

import json
import os
import pandas as pd

IN_PATH = "data/processed/cicids2017_stage2_clean.csv"
OUT_PATH = "data/processed/cicids2017_labeled.csv"
BINARY_MAP_PATH = "models/metadata/label_map_binary.json"
MULTICLASS_MAP_PATH = "models/metadata/label_map_multiclass.json"

# -----------------------------------------------------------------------
# Raw label -> taxonomy class mapping (Step 1.2 attack taxonomy).
# Keys are matched AFTER whitespace-stripping and encoding cleanup, so
# they don't need to account for stray spaces or mangled dash characters.
# -----------------------------------------------------------------------
TAXONOMY_MAP = {
    "BENIGN": "Benign",

    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",

    "DDoS": "DDoS",

    "PortScan": "PortScan",

    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "BruteForce",

    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",

    "Infiltration": "Infiltration",
    "Bot": "Bot",
    "Heartbleed": "Heartbleed",
}


def clean_label_column(series: pd.Series) -> pd.Series:
    """
    Fix whitespace and encoding inconsistencies in the raw Label column
    BEFORE any mapping is attempted — mapping against dirty strings would
    silently produce "unmapped" results for labels that are actually
    known, just spelled with stray characters.
    """
    # Strip leading/trailing whitespace — CIC-IDS2017 is known to have
    # inconsistent spacing in string fields across its day-files.
    cleaned = series.str.strip()

    # The mangled "�" character seen in "Web Attack � Brute Force" is a
    # decoding artifact — the source file used a dash character that
    # didn't survive whatever encoding step produced this CSV release.
    # We normalize it to a plain hyphen so it matches our TAXONOMY_MAP keys.
    cleaned = cleaned.str.replace("\ufffd", "-", regex=False)  # U+FFFD replacement character
    cleaned = cleaned.str.replace("�", "-", regex=False)        # literal mojibake fallback

    # Collapse any resulting double-spaces around the normalized hyphen
    cleaned = cleaned.str.replace(r"\s*-\s*", " - ", regex=True)

    return cleaned


def build_multiclass_target(label_col: pd.Series) -> tuple[pd.Series, list]:
    """
    Map each cleaned raw label to its taxonomy class name. Any raw label
    NOT found in TAXONOMY_MAP is flagged rather than silently dropped or
    silently binned into an existing class — an unmapped label (like the
    "Comb" value we found in our EDA) needs a deliberate decision, not an
    accidental one.
    """
    mapped = label_col.map(TAXONOMY_MAP)
    unmapped_mask = mapped.isna()
    unmapped_values = sorted(label_col[unmapped_mask].unique().tolist())

    if unmapped_values:
        print(f"WARNING: {unmapped_mask.sum():,} rows have unmapped raw labels: {unmapped_values}")
        print("These rows will be assigned to an 'Unknown' taxonomy class rather than dropped silently.")
        mapped = mapped.fillna("Unknown")

    return mapped, unmapped_values


def main():
    print(f"Loading {IN_PATH}...")
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Loaded shape: {df.shape}\n")

    label_col_name = "Label" if "Label" in df.columns else df.columns[-1]

    # Step 1 — clean the raw label text before any mapping is attempted
    df[label_col_name] = clean_label_column(df[label_col_name].astype(str))

    # -------------------------------------------------------------
    # Step 2 — binary target: Benign (0) vs Attack (1)
    # -------------------------------------------------------------
    # Built directly from the cleaned raw label, independent of the
    # taxonomy grouping, so it's unaffected by any taxonomy mapping
    # decisions (like the "Unknown" fallback above) — a flow is either
    # literally labeled BENIGN or it isn't.
    df["label_binary"] = (df[label_col_name].str.upper() != "BENIGN").astype(int)
    binary_map = {"Benign": 0, "Attack": 1}

    # -------------------------------------------------------------
    # Step 3 — multiclass target: taxonomy class, integer-encoded
    # -------------------------------------------------------------
    taxonomy_labels, unmapped = build_multiclass_target(df[label_col_name])

    # Build the class -> integer mapping in a FIXED, sorted order (not
    # insertion order) so this mapping is identical every time the script
    # runs, regardless of which classes happen to appear first in the data.
    unique_classes = sorted(taxonomy_labels.unique().tolist())
    multiclass_map = {cls: i for i, cls in enumerate(unique_classes)}
    df["label_multiclass"] = taxonomy_labels.map(multiclass_map)

    # -------------------------------------------------------------
    # Step 4 — save both mappings as JSON
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(BINARY_MAP_PATH), exist_ok=True)
    with open(BINARY_MAP_PATH, "w") as f:
        json.dump(binary_map, f, indent=2)
    with open(MULTICLASS_MAP_PATH, "w") as f:
        json.dump(multiclass_map, f, indent=2)
    print(f"\nSaved binary label map to {BINARY_MAP_PATH}: {binary_map}")
    print(f"Saved multiclass label map to {MULTICLASS_MAP_PATH}: {multiclass_map}")

    # -------------------------------------------------------------
    # Step 5 — print class distributions for both targets
    # -------------------------------------------------------------
    print("\nBinary target distribution:")
    binary_counts = df["label_binary"].value_counts().rename(index={0: "Benign", 1: "Attack"})
    print(binary_counts)

    print("\nMulticlass target distribution:")
    inv_multiclass_map = {v: k for k, v in multiclass_map.items()}
    multiclass_counts = df["label_multiclass"].value_counts().rename(index=inv_multiclass_map)
    print(multiclass_counts)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved labeled dataset to {OUT_PATH} — shape {df.shape}")

    if unmapped:
        print(f"\nACTION NEEDED: {len(unmapped)} raw label(s) were unmapped and assigned to 'Unknown': {unmapped}")
        print("Decide whether to map these explicitly in TAXONOMY_MAP, or exclude them, before training.")


if __name__ == "__main__":
    main()