import argparse
from pathlib import Path

import numpy as np
import torch
from maskclip_onnx import clip
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from project_config import (
    DEPTH_THRESHOLD,
    FUTURE_WINDOW,
    MODEL_PATHS,
    TRAIN_DEPTH_DIR,
    TRAIN_IMAGE_DIR,
    TRAIN_OTAS_MASK_FILE,
    TRAIN_ROBOT_CSV,
    TRAIN_TIMESTAMP_CSV,
    TRAV_THRESHOLD,
)

from utils.depth import depth_mask_from_file
from utils.features import extract_patch_features
from utils.masks import (
    combine_masks,
    create_padding_mask,
    create_trav_mask,
    load_otas_masks,
)
from utils.models import save_model
from utils.traversability import (
    load_timestamp_map,
    robonav_score_series,
)


def build_training_data(mask_type, model, preprocess, device):
    timestamps = load_timestamp_map(TRAIN_TIMESTAMP_CSV)

    score_df = robonav_score_series(TRAIN_ROBOT_CSV)
    score_times = score_df["time"].to_numpy()
    score_values = score_df["trav_score"].to_numpy()

    padding = create_padding_mask(48, 48)

    trav = (
        create_trav_mask(48, 48)
        if mask_type == "trav"
        else None
    )

    otas = (
        load_otas_masks(TRAIN_OTAS_MASK_FILE)
        if mask_type == "otas"
        else {}
    )

    x_parts = []
    y_parts = []

    image_files = sorted(
        p
        for p in Path(TRAIN_IMAGE_DIR).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    for i, image_path in enumerate(image_files, 1):

        if image_path.name not in timestamps:
            continue

        # --------------------------------------------------
        # Ground Truth
        # --------------------------------------------------

        target_time = (
            timestamps[image_path.name]
            + FUTURE_WINDOW
        )

        score_idx = np.abs(
            score_times - target_time
        ).argmin()

        score = float(score_values[score_idx])

        label = int(
            score >= TRAV_THRESHOLD
        )

        # --------------------------------------------------
        # Additional masks
        # --------------------------------------------------

        extra_otas = (
            otas.get(image_path.name)
            if mask_type == "otas"
            else None
        )

        extra_depth = None

        if mask_type == "depth":

            depth_path = (
                TRAIN_DEPTH_DIR
                / f"{image_path.stem}.npy"
            )

            extra_depth = depth_mask_from_file(
                depth_path,
                48,
                48,
                DEPTH_THRESHOLD,
            )

        # --------------------------------------------------
        # Combine masks
        # --------------------------------------------------

        valid = combine_masks(
            padding,
            trav=trav,
            otas=extra_otas,
            depth=extra_depth,
        )

        if not valid.any():
            continue

        # --------------------------------------------------
        # MaskCLIP patch features
        # --------------------------------------------------

        patches = extract_patch_features(
            model,
            preprocess,
            str(image_path),
            device,
        )[valid]

        # Each valid patch receives the frame label
        x_parts.append(patches)

        y_parts.append(
            np.full(
                len(patches),
                label,
                dtype=np.int64,
            )
        )

        if i % 100 == 0:
            print(
                f"Processed {i}/{len(image_files)} images"
            )

    if not x_parts:
        raise RuntimeError(
            "No valid training patches were found."
        )

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)

    return x, y


def train(mask_type):

    # --------------------------------------------------
    # Load frozen MaskCLIP / CLIP feature extractor
    # --------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, preprocess = clip.load(
        "ViT-B/16",
        device=device,
    )

    model.eval()

    # --------------------------------------------------
    # Build training dataset
    # --------------------------------------------------

    print("Building training data...")

    x, y = build_training_data(
        mask_type,
        model,
        preprocess,
        device,
    )

    print()
    print(f"Mask:      {mask_type}")
    print(f"Patches:   {len(y)}")
    print(f"Features:  {x.shape[1]}")
    print(
        f"Labels:    "
        f"non-trav={np.sum(y == 0)}, "
        f"trav={np.sum(y == 1)}"
    )

    # --------------------------------------------------
    # Standardize features
    # --------------------------------------------------

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)

    # --------------------------------------------------
    # Train Logistic Regression
    # --------------------------------------------------

    print()
    print("Training Logistic Regression...")

    classifier = LogisticRegression(
        max_iter=1000
    )

    classifier.fit(
        x_scaled,
        y,
    )

    # --------------------------------------------------
    # Save classifier + scaler
    # --------------------------------------------------

    save_model(
        MODEL_PATHS[mask_type],
        classifier,
        scaler,
    )

    print()
    print(
        f"Saved model: "
        f"{MODEL_PATHS[mask_type]}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask",
        choices=[
            "trav",
            "otas",
            "depth",
        ],
        required=True,
    )

    args = parser.parse_args()

    train(args.mask)