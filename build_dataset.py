import os
import numpy as np

DATASET_DIR = "dataset"
OUTFILE = "dataset.npz"

labels = sorted(os.listdir(DATASET_DIR))  # expects dog, baby, shaq, etc.

X = []
y = []
label_to_idx = {label: idx for idx, label in enumerate(labels)}

print("Detected gesture folders:", labels)

for label in labels:
    folder = os.path.join(DATASET_DIR, label)
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    print(f"{label}: {len(files)} samples")

    for f in files:
        path = os.path.join(folder, f)
        X.append(np.load(path))
        y.append(label_to_idx[label])

X = np.stack(X, axis=0)
y = np.array(y)

np.savez(OUTFILE, X=X, y=y, labels=labels)
print(f"Saved {OUTFILE} with {X.shape[0]} samples and {X.shape[1]} dimensions per sample.")
