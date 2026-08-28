import argparse
from pathlib import Path
import sys

sys.path.append("otas/src")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference import single_inference
from project_config import ROBONAV_PREPROCESSING
from otas_utils import extract_otas_masks


def parse_args():
    parser = argparse.ArgumentParser(description="Generate OTAS masks for RoboNav images.")
    parser.add_argument("--dataset", choices=ROBONAV_PREPROCESSING.keys(), required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--debug", action="store_true", help="Process only 10 images.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "spot" and args.split == "train":
        raise ValueError("SPOT is configured as a test-only dataset; use --split test.")
    base_dir = ROBONAV_PREPROCESSING[args.dataset]["base_dir"]
    model = single_inference(
        enable_mask_refinement=False,
        n_components=24,
        n_clusters=18,
        dinov2_input_size=518,
        shared_feat_resolution=74,
        dino_scale_factor=2,
    )
    print(f"Dataset: {args.dataset} | split: {args.split}")
    extract_otas_masks(
        model,
        base_dir / f"images/{args.split}/left_224",
        base_dir / f"OTAS_masked/otas_masks_{args.split}.pt",
        debug_dir=base_dir / f"OTAS_masked/debug_{args.split}",
        limit=10 if args.debug else None,
    )


if __name__ == "__main__":
    main()
