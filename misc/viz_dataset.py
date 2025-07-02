import argparse
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # for isometric rendering
try:
    from skimage.measure import marching_cubes
except ImportError:  # skimage might be optional
    marching_cubes = None

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
        help="Optional title template for the figure. Available placeholders: {t} (timestep), {g} (group), {proj} (volume projection mode), {slice} (slice index, if applicable). Example: 'Group {g} – t={t} – {proj}{slice}'.")
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
    # 3-D visualisation options
    parser.add_argument(
        "--volume-proj", choices=["slice", "max", "iso"], default="slice",
        help="Method to project 3-D volumes onto 2-D for plotting. 'slice' shows a single slice (see --slice-index); 'max' shows a maximum-intensity projection along the first volume axis; 'iso' shows an isometric view of the volume.")
    parser.add_argument(
        "--slice-index", type=int, default=None,
        help="Index of the slice to show when --volume-proj=slice. Defaults to the middle slice if not specified.")
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
        if arr.ndim < 2:
            raise ValueError(f"Dataset {dset_name} must have at least 2 dimensions (time + data): {arr.shape}")
        if time_dim is None:
            time_dim = arr.shape[0]
        elif arr.shape[0] != time_dim:
            raise ValueError(
                f"Dataset {dset_name} has inconsistent time dimension {arr.shape[0]} (expected {time_dim})")
        data[dset_name] = arr
    return data, time_dim


def plot_timestep(datasets: dict, t: int, output_path: Path, cmap_scalar: str,
                  title_tpl: str | None = None, group_name: str | None = None,
                  volume_proj: str = "slice", slice_index: int | None = None):
    """Create a figure for timestep *t* containing one subplot per channel across
    all datasets. Datasets with multiple channels are laid out in rows."""
    # Determine how many subplots are needed, accounting for possible absence of channel dim
    def n_channels(arr: np.ndarray) -> int:
        """Return the number of visualisable channels for *arr*.

        Heuristic:
        • ndim ≥ 5 → assume layout (T,C,…) – return C.
        • ndim == 4:
            – If second dim ≤ 3 treat it as channel count (typical for vector
              fields such as velocity with 2 or 3 components).
            – Otherwise treat it as a depth dimension of a 3-D volume (no
              channel) and return 1.
        • everything else → 1 channel.
        """
        if arr.ndim >= 5:
            return arr.shape[1]
        if arr.ndim == 4:
            return arr.shape[1] if arr.shape[1] <= 3 else 1
        return 1

    n_plots = sum(n_channels(arr) for arr in datasets.values())
    ncols = int(np.ceil(np.sqrt(n_plots)))  # square-ish layout
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes_iter = iter(axes.flat)
    plot_counter = 0  # keep track of subplot index for potential 3-D axes

    for dset_name, arr in datasets.items():
        n_ch = n_channels(arr)
        for ch in range(n_ch):
            ax = next(axes_iter)
            plot_counter += 1
            # Extract the frame/volume for plotting
            if arr.ndim >= 5:
                img = arr[t, ch]
            elif arr.ndim == 4:
                if arr.shape[1] <= 3:
                    img = arr[t, ch]
                else:
                    img = arr[t]
            elif arr.ndim >= 3:
                # (T, C?, H, W) or (T, H, W)
                if n_ch > 1:
                    img = arr[t, ch]
                else:
                    img = arr[t]
            else:
                img = arr[t]

            # Choose plotting method based on dimensionality
            if volume_proj == "iso" and img.ndim == 3:
                # Produce 3-D isometric plot of the volume
                # Convert current 2-D axis to 3-D
                ax.remove()
                ax = fig.add_subplot(nrows, ncols, plot_counter, projection="3d")
                # Use marching cubes if available, else fallback to voxel plot
                try:
                    cmap_sel = "coolwarm" if dset_name.lower().startswith("velocity") or (n_ch >= 2 and dset_name.lower().startswith("vel")) else cmap_scalar
                    cmap_obj = plt.get_cmap(cmap_sel)
                    vmin, vmax = float(img.min()), float(img.max())

                    if marching_cubes is not None:
                        verts, faces, normals, verts_intensity = marching_cubes(img, level=float(np.mean(img)))
                        mesh = Poly3DCollection(verts[faces], alpha=0.7)
                        if verts_intensity is None or len(verts_intensity) == 0:
                            # derive vertex intensities by sampling original volume
                            idx = np.clip(np.round(verts).astype(int), 0, np.array(img.shape) - 1)
                            verts_intensity = img[idx[:, 0], idx[:, 1], idx[:, 2]]

                        face_vals = verts_intensity[faces].mean(axis=1)
                        colors = cmap_obj((face_vals - vmin) / (vmax - vmin + 1e-8))
                        mesh.set_facecolors(colors)
                        ax.add_collection3d(mesh)
                        ax.set_xlim(0, img.shape[0])
                        ax.set_ylim(0, img.shape[1])
                        ax.set_zlim(0, img.shape[2])

                        # Draw bounding box
                        dx, dy, dz = img.shape
                        corners = [
                            (0, 0, 0), (dx, 0, 0), (dx, dy, 0), (0, dy, 0),
                            (0, 0, dz), (dx, 0, dz), (dx, dy, dz), (0, dy, dz)
                        ]
                        edge_idx = [
                            (0, 1), (1, 2), (2, 3), (3, 0),
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]
                        for i, j in edge_idx:
                            xs = [corners[i][0], corners[j][0]]
                            ys = [corners[i][1], corners[j][1]]
                            zs = [corners[i][2], corners[j][2]]
                            ax.plot(xs, ys, zs, color="black", linewidth=0.5)
                    else:
                        # Fallback rough voxel rendering
                        filled = img > np.mean(img)
                        img_norm = (img - vmin) / (vmax - vmin + 1e-8)
                        voxel_colors = cmap_obj(img_norm)
                        ax.voxels(filled, facecolors=voxel_colors, edgecolor="k")
                    ax.set_axis_off()

                    # Add colorbar for iso view
                    from matplotlib.cm import ScalarMappable
                    sm = ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                    sm.set_array([])
                    fig.colorbar(sm, ax=ax, shrink=0.7)
                except Exception as e:
                    ax.text(0.5, 0.5, 0.5, f"ISO err: {e}", ha="center")
                im_data = None  # nothing for 2-D processing further

            elif img.ndim == 3:  # 3-D volume -> project to 2-D
                if volume_proj == "max":
                    img2d = img.max(axis=0)
                else:  # 'slice'
                    idx = slice_index if slice_index is not None else img.shape[0] // 2
                    idx = max(0, min(idx, img.shape[0] - 1))
                    img2d = img[idx]
                im_data = img2d
            else:
                im_data = img

            if im_data is not None and im_data.ndim == 2:
                # 2D field -> image
                if dset_name.lower().startswith("velocity") or (n_ch >= 2 and dset_name.lower().startswith("vel")):
                    im = ax.imshow(im_data, cmap="coolwarm", origin="lower")
                else:
                    im = ax.imshow(im_data, cmap=cmap_scalar, origin="lower")
                fig.colorbar(im, ax=ax, shrink=0.7)
            elif im_data is not None and im_data.ndim == 1:
                ax.plot(im_data)
            else:
                if im_data is not None:
                    raise ValueError(f"Unsupported data dimensionality for plotting: {im_data.shape}")

            # Title per subplot
            if n_ch > 1:
                ax.set_title(f"{dset_name} [ch {ch}] t={t}")
            else:
                ax.set_title(f"{dset_name} t={t}")

            # Hide axes for images; keep for line plots
            if im_data is not None and im_data.ndim == 2:
                ax.axis("off")

    # Hide any remaining unused axes
    for ax in axes_iter:
        ax.axis("off")

    # Build formatting dictionary for titles
    title_vars = {"t": t, "proj": volume_proj, "slice": slice_index if slice_index is not None else ""}
    if group_name is not None:
        title_vars["g"] = group_name

    if title_tpl:
        fig.suptitle(title_tpl.format(**title_vars), fontsize=16)
    else:
        # Default title showing group & timestep only
        default_title = f"{group_name or ''}"
        fig.suptitle(default_title.strip(), fontsize=16)

    # Projection details placed just below the main title
    proj_descr = f"{volume_proj}" if volume_proj != "slice" else f"slice {title_vars['slice']}"
    fig.text(0.5, 0.92, proj_descr, ha="center", va="top", fontsize=10, style="italic")

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
                plot_timestep(
                    datasets, t, out_file,
                    cmap_scalar=args.cmap,
                    title_tpl=args.title,
                    group_name=group_name,
                    volume_proj=args.volume_proj,
                    slice_index=args.slice_index,
                )
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