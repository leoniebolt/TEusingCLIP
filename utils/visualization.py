from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from project_config import GRID_SIZE


def smooth_heatmap(values, valid_mask, target_shape):
    heatmap = values.reshape(GRID_SIZE, GRID_SIZE)
    mask = valid_mask.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32)
    h, w = target_shape
    filled = np.nan_to_num(heatmap, nan=0.0)
    resized_values = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
    blurred_values = ndimage.gaussian_filter(resized_values, sigma=0.5, mode="nearest")
    blurred_mask = ndimage.gaussian_filter(resized_mask, sigma=0.5, mode="nearest")
    smooth = blurred_values / np.where(blurred_mask == 0, 1e-5, blurred_mask)
    return np.clip(smooth, 0.0, 1.0), np.clip(blurred_mask * 1.5, 0.0, 1.0)


def save_overlay(image_rgb, values, valid_mask, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap, alpha = smooth_heatmap(values, valid_mask, image_rgb.shape[:2])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image_rgb)
    ax.imshow(heatmap, cmap="RdYlGn", alpha=alpha * 0.6, vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
