from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def similarity_to_patch_mask(similarity, grid_size=14, threshold=0.5):
    similarity = torch.as_tensor(similarity, dtype=torch.float32).squeeze()
    valid_score = 1.0 - similarity
    grid = F.interpolate(
        valid_score[None, None], size=(grid_size, grid_size), mode="bilinear", align_corners=False
    ).squeeze()
    return valid_score, grid >= threshold


def save_mask_overlay(image, patch_mask, output_path):
    image_np = np.asarray(image.convert("RGB"))
    h, w = image_np.shape[:2]
    mask = patch_mask.detach().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = image_np.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[~mask] = (0.5 * overlay[~mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


def extract_otas_masks(model, image_dir, output_file, debug_dir=None, raw_dir=None,
                       pos_prompts=("sky", "tree"), neg_prompts=("path", "grounds"), limit=None):
    image_dir, output_file = Path(image_dir), Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None:
        images = images[:limit]
    print(f"[OTAS] Images found: {len(images)}")

    dataset = []
    for i, image_path in enumerate(images, 1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            similarity = model.similarity_single(image, pos_prompts=list(pos_prompts), neg_prompts=list(neg_prompts))
            valid_score, patch_mask = similarity_to_patch_mask(similarity)

            dataset.append({"frame_id": image_path.name, "otas_mask_1d": patch_mask.flatten().cpu()})
            if raw_dir:
                np.save(raw_dir / f"{image_path.stem}_maskrefinement.npy", to_numpy(valid_score))
            if debug_dir:
                save_mask_overlay(image, patch_mask, debug_dir / f"debug_{image_path.name}")

        if i % 100 == 0 or i == len(images):
            print(f"[OTAS] Processed {i}/{len(images)}")

    torch.save(dataset, output_file)
    print(f"[OTAS] Saved: {output_file}")
