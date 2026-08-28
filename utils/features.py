import numpy as np
import torch
from PIL import Image


def extract_patch_features(model, preprocess, image_or_path, device):
    if isinstance(image_or_path, (str, bytes)):
        image = Image.open(image_or_path).convert("RGB")
    else:
        image = image_or_path.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        visual = model.visual
        x = visual.conv1(image_tensor.type(model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        x = visual.transformer(x).permute(1, 0, 2)
        patches = visual.ln_post(x[:, 1:, :])
        if visual.proj is not None:
            patches = patches @ visual.proj
        patches = patches / patches.norm(dim=-1, keepdim=True)
    return patches.squeeze(0).cpu().numpy().astype(np.float32)
