"""Group-wise dimensionality-reduction visualizer.

The script loads one summary vector per HDF5 *group* from separate train and
test files, then projects those vectors to 2-D using PCA and UMAP.  Two
side-by-side scatter plots (PCA and UMAP) colour points by dataset split and
annotate them with their original group names.

Usage (defaults shown):
    python groupwise_reduced_dim.py \
        --train_h5 ./data/fluids/Droplet_Contact/2D/train.h5 \
        --test_h5  ./data/fluids/Droplet_Contact/2D/test.h5
"""

import argparse
import h5py
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

def summarize_group(group):
    """Extracts a summary vector for one HDF5 group."""
    field_summaries = []
    for field_name in group.keys():
        data = group[field_name][()]  # shape: (T, C, H, W)
        avg = data.mean(axis=(0, 2, 3))  # shape: (C,) (average over time, height, width)
        field_summaries.append(avg)
    return np.concatenate(field_summaries)

def load_group_vectors(h5_file):
    with h5py.File(h5_file, 'r') as f:
        group_names = list(f.keys())
        vectors = []
        for g in tqdm(group_names, desc=f"Reading groups from {Path(h5_file).name}"):
            summary = summarize_group(f[g])
            vectors.append(summary)
    return np.array(vectors), group_names

# ------------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------------
default_train = "./data/fluids/KS/2D/train.h5"
default_test = "./data/fluids/KS/2D/test.h5"

parser = argparse.ArgumentParser(
    description="Plot PCA and UMAP projections of group-averaged channel data from train/test HDF5 files."
)
parser.add_argument("--train_h5", type=str, default=default_train, help="Path to the train .h5 file")
parser.add_argument("--test_h5", type=str, default=default_test, help="Path to the test .h5 file")
parser.add_argument("--out_png", type=str, default=None, help="Filename for saved PNG (auto if omitted)")

args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
# Output file name
# ------------------------------------------------------------------
if args.out_png is not None:
    out_png = args.out_png
else:
    train_path = Path(args.train_h5)
    try:
        dataset_name = train_path.parents[1].name  # e.g., KS
        dim_tag = train_path.parent.name  # 1D, 2D
    except IndexError:
        dataset_name = train_path.stem
        dim_tag = "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = f"groupwise_reduced_dim_{dataset_name}_{dim_tag}_{timestamp}.png"

# ---- Load from train.h5 and test.h5 ----
X_train, group_train = load_group_vectors(args.train_h5)
X_test, group_test = load_group_vectors(args.test_h5)

X_all = np.concatenate([X_train, X_test], axis=0)
group_all = group_train + group_test
source_all = ['train'] * len(group_train) + ['test'] * len(group_test)

# ---- Dimensionality Reduction ----
# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all)

# UMAP
reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
X_umap = reducer.fit_transform(X_all)  # shape: (N_groups, 2)

# ---- Plot side-by-side ----
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
methods = {
    "PCA": X_pca,
    "UMAP": X_umap,
}
colors = {"train": "tab:blue", "test": "tab:orange"}

for ax, (method_name, emb) in tqdm(list(zip(axs, methods.items())), desc="Plotting"):
    for src in ["train", "test"]:
        idx = [i for i, s in enumerate(source_all) if s == src]
        ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=60,
            alpha=0.7,
            label=src.upper(),
            color=colors[src],
            edgecolors="k",
        )

    # Annotate group names
    for i, name in enumerate(group_all):
        ax.text(
            emb[i, 0],
            emb[i, 1],
            name.replace("Re_", ""),
            fontsize=7,
            ha="center",
            va="center",
            alpha=0.6,
        )

    ax.set_title(f"Group-wise {method_name} (Train vs Test)")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.grid(True)

# Combine legend from the last axis
axs[-1].legend()
fig.tight_layout()
#plt.show()
fig.savefig(out_png, bbox_inches="tight")
print(f"Figure saved to {out_png}")