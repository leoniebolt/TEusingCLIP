import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from maskclip_onnx import clip
from project_config import DATASETS, DEPTH_THRESHOLD, GRID_SIZE, MODEL_PATHS, OUTPUT_DIRS
from utils.depth import depth_mask_from_file
from utils.evaluation import binary_metrics, save_confusion_matrix
from utils.features import extract_patch_features
from utils.masks import combine_masks, create_padding_mask, load_otas_masks
from utils.models import load_model
from utils.traversability import rellis_gt_map, robonav_gt_map
from utils.visualization import save_overlay


def ground_truth(config):
    if config.name == "rellis":
        return rellis_gt_map(config.imu_csv, config.odom_csv, config.timestamp_csv)
    return robonav_gt_map(config.robot_csv, config.timestamp_csv)


def test(dataset_name, mask_type, save_heatmaps=True):
    config = DATASETS[dataset_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_model, preprocess = clip.load("ViT-B/16", device=device)
    feature_model.eval()
    classifier, scaler = load_model(MODEL_PATHS[mask_type])
    gt = ground_truth(config)
    padding = create_padding_mask(config.pad_top, config.pad_bottom)
    otas = load_otas_masks(config.otas_mask_file) if mask_type == "otas" else {}
    output_dir = OUTPUT_DIRS[(dataset_name, mask_type)]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions, all_targets = [], []
    image_files = sorted(p for p in config.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for i, image_path in enumerate(image_files, 1):
        otas_mask = otas.get(image_path.name) if mask_type == "otas" else None
        depth_mask = None
        if mask_type == "depth":
            depth_path = config.depth_dir / f"{image_path.stem}.npy"
            depth_mask = depth_mask_from_file(depth_path, config.pad_top, config.pad_bottom, DEPTH_THRESHOLD)
        valid = combine_masks(padding, otas=otas_mask, depth=depth_mask)
        values = np.full(GRID_SIZE * GRID_SIZE, np.nan, dtype=np.float32)
        if valid.any():
            features = extract_patch_features(feature_model, preprocess, str(image_path), device)
            probabilities = classifier.predict_proba(scaler.transform(features[valid]))[:, 1]
            values[valid] = probabilities
            score = gt.get(image_path.name, gt.get(image_path.stem))
            if score is not None:
                all_predictions.extend(probabilities.tolist())
                all_targets.extend([score] * len(probabilities))
        if save_heatmaps:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            save_overlay(image, values, valid, output_dir / f"{image_path.stem}_heatmap.png")
        if i % 100 == 0:
            print(f"Processed {i}/{len(image_files)} images")
    metrics = binary_metrics(all_predictions, all_targets)
    print(f"Dataset: {dataset_name}")
    print(f"Mask:    {mask_type}")
    print(f"Samples: {metrics['n']}")
    print(f"F1:      {metrics['f1']:.4f}")
    print(f"MCC:     {metrics['mcc']:.4f}")
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(metrics["confusion_matrix"])
    save_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mattro", "spot", "rellis"], required=True)
    parser.add_argument("--mask", choices=["trav", "otas", "depth"], required=True)
    parser.add_argument("--no-heatmaps", action="store_true")
    args = parser.parse_args()
    test(args.dataset, args.mask, save_heatmaps=not args.no_heatmaps)
