import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_config import DEPTH_THRESHOLD, ROBONAV_PREPROCESSING
from depth_utils import generate_depth_maps


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RoboNav stereo depth maps.")
    parser.add_argument("--dataset", choices=ROBONAV_PREPROCESSING.keys(), required=True)
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ROBONAV_PREPROCESSING[args.dataset]
    base_dir = cfg["base_dir"]
    print(f"Dataset: {args.dataset} | max depth: {DEPTH_THRESHOLD} m")
    generate_depth_maps(
        base_dir / "images/rectified/left",
        base_dir / "images/rectified/right",
        base_dir / "depth_masked/depth_information",
        cfg["depth_fx"],
        cfg["depth_baseline"],
        max_depth=DEPTH_THRESHOLD,
        save_visualizations=not args.no_visualizations,
        channels=3,
    )


if __name__ == "__main__":
    main()
