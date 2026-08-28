import numpy as np
import torch
from project_config import GRID_SIZE, PATCH_SIZE, ROI_ROWS, ROI_COLS


def create_padding_mask(pad_top, pad_bottom):
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    first = pad_top // PATCH_SIZE
    last = GRID_SIZE - (pad_bottom // PATCH_SIZE)
    mask[first:last, :] = True
    return mask.reshape(-1)


def create_trav_mask(pad_top, pad_bottom, rows=ROI_ROWS, cols=ROI_COLS):
    padding = create_padding_mask(pad_top, pad_bottom).reshape(GRID_SIZE, GRID_SIZE)
    mask = np.zeros_like(padding)
    r0, r1 = rows
    c0, c1 = cols
    mask[r0:r1, c0:c1] = True
    return (mask & padding).reshape(-1)


def load_otas_masks(path):
    data = torch.load(path, weights_only=False)
    result = {}
    for entry in data:
        key = entry.get("frame_id", entry.get("filename"))
        value = entry.get("otas_mask_1d", entry.get("otas_mask"))
        if key is None or value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        result[str(key)] = np.asarray(value).astype(bool).reshape(-1)
    return result


def combine_masks(padding, trav=None, otas=None, depth=None):
    valid = np.asarray(padding, dtype=bool).copy()
    for mask in (trav, otas, depth):
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
    return valid
