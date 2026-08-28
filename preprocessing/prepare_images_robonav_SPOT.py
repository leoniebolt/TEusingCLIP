from pathlib import Path
import pandas as pd
from image_utils import paired_image_names, process_image_pairs

DATASET_DIR = Path("datasets/robonav/spot_route1a")
TARGET_SIZE = 224
CSV_FILE = DATASET_DIR / "csv_files/timestamps_left.csv"
LEFT_DIR = DATASET_DIR / "images/rectified/left"
RIGHT_DIR = DATASET_DIR / "images/rectified/right"


def find_filename_column(df):
    for name in ("filename", "image", "image_name", "name"):
        if name in df.columns:
            return name
    raise ValueError(f"No filename column found. Available columns: {df.columns.tolist()}")


def main():
    df = pd.read_csv(CSV_FILE)
    filename_col = find_filename_column(df)
    valid_names = set(paired_image_names(LEFT_DIR, RIGHT_DIR))
    test_df = df[df[filename_col].isin(valid_names)].reset_index(drop=True)
    test_df.to_csv(DATASET_DIR / "csv_files/test.csv", index=False)
    print(f"Usable image pairs: {len(test_df)}")

    process_image_pairs(
        test_df[filename_col].tolist(), LEFT_DIR, RIGHT_DIR,
        DATASET_DIR / "images/test/left_224",
        DATASET_DIR / "images/test/right_224",
        TARGET_SIZE,
    )


if __name__ == "__main__":
    main()
