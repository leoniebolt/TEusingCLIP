from pathlib import Path
from depth_utils import generate_depth_maps

BASE_DIR = Path("datasets/rellis_3d/00000_00")
LEFT_DIR = BASE_DIR / "images/original/images_left"
RIGHT_DIR = BASE_DIR / "images/original/images_right"
DEPTH_DIR = BASE_DIR / "depth_information"
FX = 614.127436
BASELINE = 0.25061983
MAX_DEPTH = 5.0
SAVE_VISUALIZATIONS = True


if __name__ == "__main__":
    generate_depth_maps(
        LEFT_DIR, RIGHT_DIR, DEPTH_DIR, FX, BASELINE,
        max_depth=MAX_DEPTH, save_visualizations=SAVE_VISUALIZATIONS, channels=1,
    )
