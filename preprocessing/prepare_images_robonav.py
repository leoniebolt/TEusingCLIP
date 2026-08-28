from pathlib import Path
import numpy as np
import pandas as pd
from image_utils import paired_image_names, process_image_pairs

DATASET_DIR = Path("datasets/robonav/mattro_route1")
TARGET_SIZE = 224
RANDOM_SEED = 29
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
    df = df[df[filename_col].isin(valid_names)].reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(df))
    split = len(indices) // 2
    train_df = df.iloc[indices[:split]].copy()
    test_df = df.iloc[indices[split:]].copy()

    csv_dir = DATASET_DIR / "csv_files"
    train_df.to_csv(csv_dir / "train.csv", index=False)
    test_df.to_csv(csv_dir / "test.csv", index=False)
    print(f"Usable image pairs: {len(df)} | train: {len(train_df)} | test: {len(test_df)}")

    for split_name, split_df in (("train", train_df), ("test", test_df)):
        process_image_pairs(
            split_df[filename_col].tolist(), LEFT_DIR, RIGHT_DIR,
            DATASET_DIR / f"images/{split_name}/left_224",
            DATASET_DIR / f"images/{split_name}/right_224",
            TARGET_SIZE,
        )


if __name__ == "__main__":
    main()
