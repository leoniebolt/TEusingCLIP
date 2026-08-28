import sys
from pathlib import Path

sys.path.append("otas/src")
from inference import single_inference
from otas_utils import extract_otas_masks

BASE_DIR = Path("datasets/rellis_3d/00000_00")
DEBUG_ONLY = False


def main():
    model = single_inference(
        enable_mask_refinement=False,
        n_components=24,
        n_clusters=18,
        dinov2_input_size=518,
        shared_feat_resolution=74,
        dino_scale_factor=2,
    )
    extract_otas_masks(
        model,
        BASE_DIR / "images/test/left_224",
        BASE_DIR / "otas_masks.pt",
        debug_dir=BASE_DIR / "images/otas_masked_images",
        raw_dir=BASE_DIR / "otas_masks",
        limit=10 if DEBUG_ONLY else None,
    )


if __name__ == "__main__":
    main()
