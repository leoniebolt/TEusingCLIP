import cv2
import numpy as np
from project_config import GRID_SIZE, IMAGE_SIZE, DEPTH_THRESHOLD


def resize_and_pad_depth(depth_map, pad_top, pad_bottom, target_size=IMAGE_SIZE):
    valid_height = target_size - pad_top - pad_bottom
    h, w = depth_map.shape[:2]
    scale = min(target_size / w, valid_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((target_size, target_size), np.nan, dtype=np.float32)
    x0 = (target_size - new_w) // 2
    y0 = pad_top + (valid_height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def patch_depths(depth_map, grid_size=GRID_SIZE):
    h, w = depth_map.shape[:2]
    ph, pw = h // grid_size, w // grid_size
    out = np.full(grid_size * grid_size, np.nan, dtype=np.float32)
    k = 0
    for r in range(grid_size):
        for c in range(grid_size):
            patch = depth_map[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            values = patch[np.isfinite(patch)]
            if values.size:
                out[k] = np.nanmedian(values)
            k += 1
    return out


def depth_mask_from_file(path, pad_top, pad_bottom, threshold=DEPTH_THRESHOLD):
    if not path.exists():
        return np.zeros(GRID_SIZE * GRID_SIZE, dtype=bool)
    depth = np.load(path)
    aligned = resize_and_pad_depth(depth, pad_top, pad_bottom)
    values = patch_depths(aligned)
    return np.isfinite(values) & (values <= threshold)
