# In der OTAS Conda-Umgebung ausführen!
import sys; sys.path.append("otas/src")
from inference import single_inference
from PIL import Image
import os
import torch
import torch.nn.functional as F

# CHANGE
#base_dir = "datasets/robonav/mattro_route1"
base_dir = "datasets/robonav/spot_route1a"
image_dir = os.path.join(base_dir, "images/images_224/left")

# DO NOT CHANGE
output_dir = base_dir
otas_mask_file = os.path.join(output_dir, "otas_masks_test.pt")

def extract_and_save_otas():
    model = single_inference(
        enable_mask_refinement=False,
        n_components=24,
        n_clusters=18,
        dinov2_input_size=518,
        shared_feat_resolution=74,
        dino_scale_factor=2
    )
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"[OTAS] Images found: {len(image_files)}")

    otas_dataset = []

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # OTAS Inferenz
        res = model.similarity_single(img, pos_prompts=["sky", "tree"], neg_prompts=["path", "grounds"])
        res = 1 - res  # Heatmap: Hohe Werte = Hindernis/Nicht-befahrbar

        # Rescale / Downsample auf 14x14 Patch-Grid-Ebene (passend zu MaskCLIP ViT-B/16)
        if not isinstance(res, torch.Tensor):
            res_tensor = torch.tensor(res, dtype=torch.float32)
        else:
            res_tensor = res.float()

        # Reshape auf 14x14 Grid via Max-Pooling oder Interpolation
        # res_tensor: [224, 224] -> [1, 1, 224, 224] -> [1, 1, 14, 14]
        res_grid = F.interpolate(res_tensor.unsqueeze(0).unsqueeze(0), size=(14, 14), mode='bilinear')
        res_grid = res_grid.squeeze().flatten()  # Shape: [196]

        # Schwellenwert anwenden für binäre Maske (True = Gültig/Frei, False = Blockiert von OTAS)
        # Bsp: Wenn res_grid < 0.5 ist, ist es KEIN Hindernis (Gültiger Patch)
        otas_valid_patch_mask = (res_grid < 0.5)

        otas_dataset.append({
            "frame_id": img_name,
            "otas_mask_1d": otas_valid_patch_mask.cpu() # 196 Bool values
        })

        if (idx + 1) % 100 == 0:
            print(f"[OTAS] Processed: {idx + 1} / {len(image_files)}")

    torch.save(otas_dataset, otas_mask_file)
    print(f"✅ OTAS Masks successfully saved to: {otas_mask_file}")

if __name__ == "__main__":
    extract_and_save_otas()