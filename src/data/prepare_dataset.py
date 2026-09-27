import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

RAW_PATH = "data/raw/CIC-IDS-2017-V2.csv"
OUT_PATH = "data/processed/cicids2017_clean.csv"

# 1. Load the dataset — this file is ~1.85GB, so this step alone may take a minute or two
print("Loading dataset...")
data = pd.read_csv(RAW_PATH, low_memory=False)
data.columns = data.columns.str.strip()  # CIC-IDS columns often have leading spaces
print(f"Loaded shape: {data.shape}")
#
# # 2. Drop leakage columns — these let the model "cheat" instead of learning real patterns
leakage_cols = ["Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
                 "Source Port", "Src Port", "Timestamp"]
data = data.drop(columns=[c for c in leakage_cols if c in data.columns], errors="ignore")

# # 3. Replace infinite values with NaN, then drop rows with any NaN
data = data.replace([np.inf, -np.inf], np.nan)
before = len(data)
data = data.dropna()
print(f"Dropped {before - len(data)} rows with inf/NaN")

# # 4. Drop exact duplicate rows
before = len(data)
data = data.drop_duplicates()
print(f"Dropped {before - len(data)} duplicate rows")
#
# 5. Drop constant columns — they carry zero information for the model
nunique = data.nunique()
constant_cols = nunique[nunique <= 1].index.tolist()
data = data.drop(columns=constant_cols)
print(f"Dropped {len(constant_cols)} constant columns: {constant_cols}")

# 6. Show class distribution
label_col = "Label" if "Label" in data.columns else data.columns[-1]
print("\nClass distribution:")
print(data[label_col].value_counts())

# 7. Save cleaned dataset
os.makedirs("data/processed", exist_ok=True)
data.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned dataset: {OUT_PATH} — final shape {data.shape}")
import matplotlib.pyplot as plt

counts = data[label_col].value_counts()
plt.figure(figsize=(10, 6))
counts.plot(kind="barh", color="#1a3c6e")
plt.xscale("log")  # log scale because Benign dwarfs everything else
plt.xlabel("Number of flows (log scale)")
plt.title("CIC-IDS2017 Class Distribution After Cleaning")
plt.tight_layout()
plt.savefig("docs/architecture/class_distribution.png", dpi=150)
print("Saved chart to docs/architecture/class_distribution.png")