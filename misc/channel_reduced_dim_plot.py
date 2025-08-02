"""Reduced-dimensionality visualization and distance analysis.

NOTE: Preferably use this script when the dataset is small! 

This script loads the data from train/test HDF5 files, flattens each channel,
projects the data to 2-D using PCA, t-SNE and UMAP, and plots the embeddings
with group colours.  It also reports the
Wasserstein distance between the train and test distributions on both PCA and
UMAP embeddings.  Pass alternative file paths via --train_h5 and --test_h5.

Example usage:
python misc/reduced_dimensionality_plots.py --train_h5 ./data/fluids/KVS_trimmed/2D/train.h5 --test_h5 ./data/fluids/KVS_trimmed/2D/test.h5
"""
import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import argparse
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

##########################################################################
#X has a list of arrays, each array is a channel
#X[0] is the first channel 
#X[0].shape is the shape of the first channel (T, H*W) which is the flattened image
#(T, H*W) is reduced to (T, 2) after PCA, t-SNE, or UMAP, 
# so each time step of each group is a point in 2D
##########################################################################

# ---- Config ----
# Default paths (can be overridden by CLI)
train_file = './data/fluids/KS/2D/train.h5'
test_file = './data/fluids/KS/2D/test.h5'

# ------------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Visualize PCA/t-SNE/UMAP embeddings and compute Wasserstein distances "
                "for train and test HDF5 datasets."
)
parser.add_argument('--train_h5', type=str, default=train_file,
                    help='Path to the train .h5 file')
parser.add_argument('--test_h5', type=str, default=test_file,
                    help='Path to the test .h5 file')
parser.add_argument('--out_png', type=str, default=None,
                    help='Filename for the saved figure (PNG). If omitted, a descriptive name is auto-generated.')
args, _ = parser.parse_known_args()  # use known to handle Jupyter or extra args gracefully

# Override defaults with CLI inputs
train_file = args.train_h5
test_file = args.test_h5

# ------------------------------------------------------------------
# Output figure filename
# ------------------------------------------------------------------
if args.out_png is not None:
    out_png = args.out_png
else:
    train_path = Path(train_file)
    try:
        dataset_name = train_path.parents[1].name  # e.g., "KS" from ./data/fluids/KS/1D/train.h5
        dim_tag = train_path.parent.name           # e.g., "1D" / "2D"
    except IndexError:
        dataset_name = train_path.stem
        dim_tag = "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = f"reduced_dim_{dataset_name}_{dim_tag}_{timestamp}.png"

# Update files list with (tag, path) tuples
files = [('train', train_file), ('test', test_file)]

X = []                # list per channel
channel_labels = []   # channel names
group_labels = []     # per sample group names (e.g., Re_100)
file_labels = []      # per sample file labels ('train' or 'test')

# ---- Discover channels from train file (assumed structure similar) ----
with h5py.File(train_file, 'r') as f:
    sample_group = next(iter(f.values()))
    discovered_channels = []
    for field in sample_group:
        shape = sample_group[field].shape  # (T, C, *spatial_dims)
        if len(shape) < 3:
            raise ValueError(f"Unexpected dataset rank {len(shape)} for field '{field}'. Expected at least 3 (T, C, ...).")
        C = shape[1]  # channel dimension is always at index 1
        for c in range(C):
            discovered_channels.append((field, c))

# ---- Accumulate data from both files ----
channel_map = {f"{field}/{c}": [] for field, c in discovered_channels}
group_map = {f"{field}/{c}": [] for field, c in discovered_channels}
file_map = {f"{field}/{c}": [] for field, c in discovered_channels}

for file_tag, filepath in tqdm(files, desc="Reading HDF5 files"):
    with h5py.File(filepath, 'r') as f:
        for group_name in f:
            group = f[group_name]
            for field, c in discovered_channels:
                data = group[field][()]  # (T, C, H, W)
                T = data.shape[0]
                flattened = data[:, c].reshape(T, -1)
                key = f"{field}/{c}"
                channel_map[key].append(flattened)
                group_map[key].extend([group_name] * T)
                file_map[key].extend([file_tag] * T)

# ---- Concatenate and prepare lists ----
for key in channel_map:
    X.append(np.concatenate(channel_map[key], axis=0))
    channel_labels.append(key)
    group_labels.append(np.array(group_map[key]))
    file_labels.append(np.array(file_map[key]))


# ---- Get all groups and assign colors for combined plot ----
# Gather all unique group names as pure Python strings, strip spaces
all_groups = set()
for glabels in group_labels:
    all_groups.update(str(g).strip() for g in glabels)
all_groups = sorted(all_groups)

cmap = pyplot.get_cmap("tab10")
# If you have more groups than colors in cmap, cycle through colors
colors = [cmap(i % cmap.N) for i in range(len(all_groups))]

group_colors = dict(zip(all_groups, colors))

# Edge colour map for train/test distinction
edge_color_map = {'train': 'black', 'test': 'white'}

# ---- Colors for train/test for second plot ----
# file_color_map = {'train': 'tab:blue', 'test': 'tab:orange'}

method_names = ['PCA', 't-SNE', 'UMAP']

# Storage for Wasserstein distances per channel
wd_results_umap = []
wd_results_pca = []

# Collect legend handles (to build a single, global legend later)
legend_handles = {}

# Precompute group to file mapping (train or test)
group_to_file = {}
for glabels, flabels in zip(group_labels, file_labels):
    for g, f in zip(glabels, flabels):
        g_str = str(g).strip()
        # Only set if not already set (first occurrence)
        if g_str not in group_to_file:
            group_to_file[g_str] = f

# ---- Plot 1: Combined groups colored by group ----
fig, axs = plt.subplots(len(X), 3, figsize=(15, 4 * len(X)))
fig.suptitle("Combined train + test groups colored by group", fontsize=16)
for i, (x, label, g_labels) in enumerate(
    tqdm(zip(X, channel_labels, group_labels), total=len(X), desc="Embedding & plotting")
):
    # ----------------------------------------------------------
    # Normalize data: zero mean & unit variance per pixel column
    # ----------------------------------------------------------
    x_scaled = StandardScaler().fit_transform(x)

    # Current file labels for this channel (needed for train/test split)
    f_labels = file_labels[i]

    embeddings = [
        PCA(n_components=2).fit_transform(x_scaled),
        TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca').fit_transform(x_scaled),
        UMAP(n_components=2).fit_transform(x_scaled)
    ]
    for j, (proj, method) in enumerate(zip(embeddings, method_names)):
        ax = axs[i, j] if len(X) > 1 else axs[j]
        # Precompute string labels once for speed
        g_labels_str = np.array([str(x).strip() for x in g_labels])

        for g in all_groups:
            g_idx = g_labels_str == g
            for src in ['train', 'test']:
                idx = g_idx & (f_labels == src)
                if not np.any(idx):
                    continue
                label_full = f"{g} ({src})"
                sc = ax.scatter(
                    proj[idx, 0],
                    proj[idx, 1],
                    s=8,
                    alpha=0.6,
                    marker='o',
                    color=group_colors[g],
                    edgecolors=edge_color_map[src],
                    linewidths=0.6,
                    label=label_full,
                )
                if label_full not in legend_handles:
                    legend_handles[label_full] = sc
        if i == 0:
            ax.set_title(method)
        if j == 0:
            ax.set_ylabel(label)
        # Keep default tick marks for better reference
        ax.grid(True)
        # Axis-level legend removed; we'll build a global legend later

    # --------------------------------------------------------------
    # Compute Wasserstein distance for UMAP embedding (index 2)
    # --------------------------------------------------------------
    umap_proj = embeddings[2]
    f_labels = file_labels[i]
    train_idx = f_labels == 'train'
    test_idx = f_labels == 'test'
    if np.any(train_idx) and np.any(test_idx):
        wd_x = wasserstein_distance(umap_proj[train_idx, 0], umap_proj[test_idx, 0])
        wd_y = wasserstein_distance(umap_proj[train_idx, 1], umap_proj[test_idx, 1])
        wd_total = np.sqrt(wd_x**2 + wd_y**2)
        wd_results_umap.append((label, wd_x, wd_y, wd_total))
    else:
        wd_results_umap.append((label, np.nan, np.nan, np.nan))

    # --------------------------------------------------------------
    # Compute Wasserstein distance for PCA embedding (index 0)
    # --------------------------------------------------------------
    pca_proj = embeddings[0]
    if np.any(train_idx) and np.any(test_idx):
        wd_x_pca = wasserstein_distance(pca_proj[train_idx, 0], pca_proj[test_idx, 0])
        wd_y_pca = wasserstein_distance(pca_proj[train_idx, 1], pca_proj[test_idx, 1])
        wd_total_pca = np.sqrt(wd_x_pca**2 + wd_y_pca**2)
        wd_results_pca.append((label, wd_x_pca, wd_y_pca, wd_total_pca))
    else:
        wd_results_pca.append((label, np.nan, np.nan, np.nan))

# Place the legend inside the figure area (right side but inside bounds)
fig.legend(
    legend_handles.values(),
    legend_handles.keys(),
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),
    fontsize="x-small",
    ncol=1,
    frameon=False,
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0, 0.78, 0.95])

# ---------------- Wasserstein distance summary -------------------
print("\nWasserstein distances between train and test (UMAP embeddings):")
for label, wd_x, wd_y, wd_total in wd_results_umap:
    print(f"{label}: WD_x={wd_x:.4f}, WD_y={wd_y:.4f}, WD_total={wd_total:.4f}")

print("\nWasserstein distances between train and test (PCA embeddings):")
for label, wd_x, wd_y, wd_total in wd_results_pca:
    print(f"{label}: WD_x={wd_x:.4f}, WD_y={wd_y:.4f}, WD_total={wd_total:.4f}")


# # ---- Plot 2: Separate train/test colors ignoring groups ----
# fig2, axs2 = plt.subplots(len(X), 3, figsize=(15, 4 * len(X)))
# fig2.suptitle("Train (blue) vs Test (orange) colored plot", fontsize=16)
# for i, (x, label, f_labels) in enumerate(zip(X, channel_labels, file_labels)):
#     embeddings = [
#         PCA(n_components=2).fit_transform(x),
#         TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca').fit_transform(x),
#         UMAP(n_components=2).fit_transform(x)
#     ]
#     for j, (proj, method) in enumerate(zip(embeddings, method_names)):
#         ax = axs2[i, j] if len(X) > 1 else axs2[j]
#         for file_tag, color in file_color_map.items():
#             idx = f_labels == file_tag
#             ax.scatter(proj[idx, 0], proj[idx, 1], s=8, alpha=0.6, label=file_tag, color=color)
#         if i == 0:
#             ax.set_title(method)
#         if j == 0:
#             ax.set_ylabel(label)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.grid(True)
#         if i == len(X) - 1 and j == 2:
#             ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0))

# plt.tight_layout(rect=[0, 0, 0.85, 0.95])
#save the plot
plt.savefig(out_png, bbox_inches="tight")
print(f"Figure saved to {out_png}")