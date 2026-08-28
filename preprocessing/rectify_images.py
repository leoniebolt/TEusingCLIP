import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_config import ROBONAV_PREPROCESSING


def camera_matrix(row):
    return np.array([[row[f"K{i}"] for i in range(0, 3)],
                     [row[f"K{i}"] for i in range(3, 6)],
                     [row[f"K{i}"] for i in range(6, 9)]], dtype=np.float64)


def distortion(row):
    return np.array([row[f"D{i}"] for i in range(5)], dtype=np.float64)


def existing_image_dir(base, side):
    candidates = [base / f"images/original/{side}", base / f"images/original/images_{side}"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No {side} image directory found below {base / 'images/original'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Rectify RoboNav stereo images.")
    parser.add_argument("--dataset", choices=ROBONAV_PREPROCESSING.keys(), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ROBONAV_PREPROCESSING[args.dataset]
    dataset_dir = cfg["base_dir"]
    baseline = cfg["rectification_baseline"]
    left_dir = existing_image_dir(dataset_dir, "left")
    right_dir = existing_image_dir(dataset_dir, "right")
    out_left = dataset_dir / "images/rectified/left"
    out_right = dataset_dir / "images/rectified/right"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    left_cam = pd.read_csv(dataset_dir / "csv_files/camera_left.csv").iloc[0]
    right_cam = pd.read_csv(dataset_dir / "csv_files/camera_right.csv").iloc[0]
    k_left, d_left = camera_matrix(left_cam), distortion(left_cam)
    k_right, d_right = camera_matrix(right_cam), distortion(right_cam)
    names = sorted({p.name for p in left_dir.glob("*.png")} & {p.name for p in right_dir.glob("*.png")})
    if not names:
        raise RuntimeError("No stereo image pairs found")

    sample = cv2.imread(str(left_dir / names[0]))
    if sample is None:
        raise RuntimeError(f"Could not read {left_dir / names[0]}")
    h, w = sample.shape[:2]
    image_size = (w, h)
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k_left, d_left, k_right, d_right, image_size,
        np.eye(3, dtype=np.float64), np.array([[-baseline], [0.0], [0.0]]),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(k_left, d_left, r1, p1, image_size, cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(k_right, d_right, r2, p2, image_size, cv2.CV_32FC1)

    print(f"Dataset: {args.dataset} | pairs: {len(names)} | baseline: {baseline} m")
    for i, name in enumerate(names, 1):
        left = cv2.imread(str(left_dir / name))
        right = cv2.imread(str(right_dir / name))
        if left is None or right is None:
            print(f"Skipping {name}: image could not be loaded")
            continue
        cv2.imwrite(str(out_left / name), cv2.remap(left, map_lx, map_ly, cv2.INTER_LINEAR))
        cv2.imwrite(str(out_right / name), cv2.remap(right, map_rx, map_ry, cv2.INTER_LINEAR))
        if i % 100 == 0 or i == len(names):
            print(f"Processed {i}/{len(names)} stereo pairs")


if __name__ == "__main__":
    main()
