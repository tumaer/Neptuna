import argparse
import json
import os
import sys
from typing import Any

"""
Example usage:
1) For saving the stats in YAML format:
python misc/compute_stats_parallel.py .../train.h5 --out-yaml <path_to_output_yaml> --log-channels Density
2) For saving the stats in JSON format:
python misc/compute_stats_parallel.py .../train.h5 --out-json <path_to_output_json> --log-channels Density
3) For printing the stats in JSON format to the terminal:
python misc/compute_stats_parallel.py .../train.h5 --log-channels Density
4) For combined statistics across multiple files:
python misc/compute_stats_parallel.py .../train.h5 .../val.h5 --out-yaml <path_to_output_yaml>
"""
# Ensure project root is importable when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.compute_stats import compute_statistics_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute HDF5 dataset statistics using compute_statistics_parallel."
    )
    parser.add_argument(
        "h5_paths",
        nargs="+",
        type=str,
        help="One or more input .h5 file paths.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Optional list of group names to include.",
    )
    parser.add_argument(
        "--frames",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Optional inclusive frame range: START END.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (default: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto).",
    )
    parser.add_argument(
        "--on-fly-stats",
        action="store_true",
        help="Run in on-the-fly mode.",
    )
    parser.add_argument(
        "--residual",
        choices=["none", "predicted", "base"],
        default="none",
        help="Residual stats mode: none | predicted | base.",
    )
    parser.add_argument(
        "--log-channels",
        nargs="+",
        default=None,
        help="Optional channel names to log-transform (e.g., pressure velocity_0).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional output path to save stats JSON.",
    )
    parser.add_argument(
        "--out-yaml",
        type=str,
        default=None,
        help=(
            "Optional output path to save stats in YAML config format "
            "(data_normalization_stats block)."
        ),
    )
    return parser.parse_args()


def build_residual_config(mode: str) -> dict[str, bool] | None:
    if mode == "none":
        return None
    if mode == "predicted":
        return {"add_predicted_value": True, "add_base_value": False}
    return {"add_predicted_value": False, "add_base_value": True}


def format_stats_as_yaml_block(
    stats: dict[str, dict[str, float]],
    root_key: str = "data_normalization_stats",
) -> str:
    lines: list[str] = [f"{root_key}:"]
    metric_order = ("mean", "std", "min", "max", "median", "iqr")

    for channel, channel_stats in stats.items():
        lines.append(f"  {channel}:")
        for metric in metric_order:
            if metric in channel_stats:
                lines.append(f"    {metric}: {channel_stats[metric]}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    h5_paths = [os.path.abspath(path) for path in args.h5_paths]
    missing_paths = [path for path in h5_paths if not os.path.isfile(path)]
    if missing_paths:
        missing_str = ", ".join(missing_paths)
        raise FileNotFoundError(f"HDF5 file(s) not found: {missing_str}")

    residual_config = build_residual_config(args.residual)

    stats, channel_names, problem_dim = compute_statistics_parallel(
        h5_paths=h5_paths,
        residual_config=residual_config,
        filter_groups=args.groups,
        filter_frames=args.frames,
        frame_stride=args.stride,
        on_fly_stats=args.on_fly_stats,
        num_workers=args.workers,
        log_transform_channels=args.log_channels,
    )

    payload: dict[str, Any] = {
        "h5_paths": h5_paths,
        "problem_dimension": problem_dim,
        "channel_names": channel_names,
        "statistics": stats,
    }

    if args.out_json is not None:
        out_path = os.path.abspath(args.out_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Saved stats to: {out_path}")

    if args.out_yaml is not None:
        yaml_path = os.path.abspath(args.out_yaml)
        yaml_text = format_stats_as_yaml_block(stats)
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        print(f"Saved YAML stats to: {yaml_path}")

    if args.out_json is None and args.out_yaml is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
