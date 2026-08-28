from pathlib import Path
from image_utils import paired_image_names, process_image_pairs

DATASET_DIR = Path("datasets/rellis_3d/00000_00")
TARGET_SIZE = 224
LEFT_DIR = DATASET_DIR / "images/original/images_left"
RIGHT_DIR = DATASET_DIR / "images/original/images_right"


def main():
    filenames = paired_image_names(LEFT_DIR, RIGHT_DIR)
    print(f"Usable image pairs: {len(filenames)}")
    process_image_pairs(
        filenames, LEFT_DIR, RIGHT_DIR,
        DATASET_DIR / "images/test/left_224",
        DATASET_DIR / "images/test/right_224",
        TARGET_SIZE,
    )


if __name__ == "__main__":
    main()
