import argparse
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

"""
Install: pip install imageio-ffmpeg 

CLI Usage:
python viz_dataset.py --h5-path data/fluids/KVS_trimmed/2D/train.h5 --step 10 --groups "<group_0>" "<group_1>"  --fps 8 --out <output_dir> --frame-range 49 99 --keep-frames --title <Figure_title>
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise all timesteps of a randomly chosen group inside a train.h5 file.")
    parser.add_argument(
        "--h5-path", dest="h5_path",
        help="Path to the .h5 file")
    parser.add_argument(
        "--out", default="trajectory_frames", help="Directory where the PNGs will be written")
    parser.add_argument(
        "--cmap", default="viridis", help="Colour-map for scalar fields; velocity channels use \"coolwarm\"")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for deterministic group selection")
    parser.add_argument(
        "--step", type=int, default=1,
        help="Save every STEP-th frame (e.g. 5 means frames 0,5,10,…). Must be ≥1.")
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Frames-per-second for the output MP4 video.")
    parser.add_argument(
        "--title", default=None,
        help="Optional title to add at the top of every generated figure. You can include '{t}' which will be replaced with the timestep index, e.g. 'Group {g} – t={t}'.")
    parser.add_argument(
        "--frame-range", nargs=2, type=int, metavar=("START", "END"),
        help="Optional inclusive range of timestep indices to process (e.g. --frame-range 10 200)."
    )
    # Mutually exclusive options for choosing which group(s) to process.  If none
    # are given we fall back to selecting a single random group.
    group_sel = parser.add_mutually_exclusive_group()
    group_sel.add_argument(
        "--groups", nargs="+", metavar="GROUP",
        help="Name(s) of one or more groups in the HDF5 file to process.")
    group_sel.add_argument(
        "--all", action="store_true",
        help="Process *all* groups found in the file. An MP4 video will be created and frame PNGs are deleted afterwards.")
    group_sel.add_argument(
        "--random", action="store_true",
        help="Explicitly request a random group to be processed (default behaviour if neither --groups nor --all is supplied).")
    parser.add_argument(
        "--keep-frames", action="store_true",
        help="If set, keep the individual PNG frame images instead of deleting them after MP4 creation.")
    return parser.parse_args()


def choose_random_group(h5file: h5py.File, rng: random.Random) -> str:
    groups = list(h5file.keys())
    if not groups:
        raise RuntimeError("HDF5 file does not contain any groups!")
    group_name = rng.choice(groups)
    print(f"Chosen group: {group_name}")
    return group_name


def load_group_datasets(group: h5py.Group):
    """Return a dict of {dataset_name: np.ndarray}. All datasets must share the
    same leading time dimension."""
    data = {}
    time_dim = None
    for dset_name, dset in group.items():
        arr = dset[...]
        if arr.ndim < 3:
            # Require at least (T, C, H, W) or (T, 1, H, W)
            raise ValueError(f"Dataset {dset_name} has fewer than 3 dimensions: {arr.shape}")
        if time_dim is None:
            time_dim = arr.shape[0]
        elif arr.shape[0] != time_dim:
            raise ValueError(
                f"Dataset {dset_name} has inconsistent time dimension {arr.shape[0]} (expected {time_dim})")
        data[dset_name] = arr
    return data, time_dim


def plot_timestep(datasets: dict, t: int, output_path: Path, cmap_scalar: str, title_tpl: str | None = None, group_name: str | None = None):
    """Create a figure for timestep *t* containing one subplot per channel across
    all datasets. Datasets with multiple channels are laid out in rows."""
    # Count total number of subplots required
    n_plots = sum(arr.shape[1] for arr in datasets.values())
    ncols = int(np.ceil(np.sqrt(n_plots)))  # square-ish layout
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes_iter = iter(axes.flat)

    for dset_name, arr in datasets.items():
        n_channels = arr.shape[1]
        for ch in range(n_channels):
            ax = next(axes_iter)
            img = arr[t, ch]
            if dset_name.lower().startswith("velocity") or (n_channels >= 2 and dset_name.lower().startswith("vel")):
                im = ax.imshow(img, cmap="coolwarm", origin="lower")
            else:
                im = ax.imshow(img, cmap=cmap_scalar, origin="lower")
            ax.set_title(f"{dset_name} [ch {ch}] t={t}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.7)

    # Hide any remaining unused axes
    for ax in axes_iter:
        ax.axis("off")

    # Optional global title
    if title_tpl:
        # Provide simple formatting variables
        fmt_kwargs = {"t": t}
        if group_name is not None:
            fmt_kwargs["g"] = group_name
        fig.suptitle(title_tpl.format(**fmt_kwargs), fontsize=16)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    h5_path = Path(args.h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Cannot find file: {h5_path}")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        # Decide which groups to process based on CLI flags
        if args.all:
            group_names = list(f.keys())
            if not group_names:
                raise RuntimeError("HDF5 file does not contain any groups!")
        elif args.groups:
            # Validate the requested group names exist in the file
            missing = [g for g in args.groups if g not in f]
            if missing:
                raise ValueError(f"Requested group(s) not found in file: {', '.join(missing)}")
            group_names = args.groups
        else:  # --random or default behaviour
            group_names = [choose_random_group(f, rng)]

        if args.step < 1:
            raise ValueError("--step must be at least 1")

        for group_name in group_names:
            datasets, n_timesteps = load_group_datasets(f[group_name])
            print(
                f"Processing group '{group_name}' containing {len(datasets)} channels, names: {list(datasets.keys())} and "
                f"{n_timesteps} timesteps. Saving every {args.step}-th frame.")

            # Determine valid timestep range
            if args.frame_range is not None:
                t_start, t_end = args.frame_range
                if t_start < 0 or t_end < 0 or t_start > t_end:
                    raise ValueError("Invalid --frame-range: START and END must be non-negative and START ≤ END")
                # Clip range to available timesteps
                t_start = max(t_start, 0)
                t_end = min(t_end, n_timesteps - 1)
            else:
                t_start, t_end = 0, n_timesteps - 1

            frame_paths = []
            for t in range(t_start, t_end + 1, args.step):
                out_file = output_dir / f"{group_name}_t{t:04d}.png"
                plot_timestep(datasets, t, out_file, cmap_scalar=args.cmap, title_tpl=args.title, group_name=group_name)
                frame_paths.append(out_file)

            # Always create an MP4 video for the processed frames
            mp4_path = output_dir / f"{group_name}.mp4"
            print(f"Creating MP4 {mp4_path} (fps={args.fps}) …")
            try:
                with imageio.get_writer(mp4_path, fps=args.fps, codec="libx264", format="ffmpeg", mode="I") as writer:
                    for frame_path in frame_paths:
                        frame = imageio.imread(frame_path)
                        frame_arr = np.asarray(frame)
                        # Ensure 3-channel RGB for all frames (grayscale→RGB, drop alpha)
                        if frame_arr.ndim == 2:  # grayscale
                            frame_arr = np.stack([frame_arr] * 3, axis=-1)
                        elif frame_arr.shape[2] == 4:  # RGBA
                            frame_arr = frame_arr[..., :3]
                        writer.append_data(frame_arr)
                print(f"MP4 saved to {mp4_path}")
            except Exception as e:
                print(f"Failed to create MP4: {e}")

            # Optionally delete intermediate PNG frames unless user asked to keep them
            if not args.keep_frames:
                for frame_path in frame_paths:
                    try:
                        frame_path.unlink()
                    except FileNotFoundError:
                        pass
                print(f"Deleted {len(frame_paths)} frame image(s) for group '{group_name}'.")
            else:
                print(f"Kept {len(frame_paths)} frame image(s) for group '{group_name}' as requested.")


if __name__ == "__main__":
    main() 