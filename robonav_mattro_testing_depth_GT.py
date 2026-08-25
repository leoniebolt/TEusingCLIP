# sources:
# https://www.geeksforgeeks.org/python/interpolation-in-python/
# https://www.geeksforgeeks.org/machine-learning/save-and-load-machine-learning-models-in-python-with-scikit-learn/

# input: model, ganzes unbekanntes Bild
# model auf unbekannten ganzen Bildern laufen lassen
# output: trav Vektor pro Bildpatch und heatmap
# Status: 100%

# Testing gesamt: 100%

import os
import pickle
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from PIL import Image
from torchvision import transforms
from maskclip_onnx import clip
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, confusion_matrix, ConfusionMatrixDisplay

device = "cuda" if torch.cuda.is_available() else "cpu"

maskclip_model, preprocess = clip.load("ViT-B/16", device=device)
maskclip_model.eval()

# Pfade
test_image_dir = "datasets/robonav/mattro_route1/images/test/images_left_224"
depth_dir = "datasets/robonav/mattro_route1/depth/depth_information"
regressor_path = "datasets/robonav/mattro_route1/depth/regressor_depth.sav"  
output_dir = "datasets/robonav/mattro_route1/depth/heatmaps_robonavmattro"
masks_dir = "datasets/robonav/mattro_route1/depth"

base_dir = "datasets/robonav/mattro_route1"

# Sensordaten & Timestamps Pfade für Ground Truth
csv_robot_path = os.path.join(base_dir, "csv_files/mattro_kantine.csv")
csv_timestamps_path = os.path.join(base_dir, "csv_files/timestamps_left.csv")

# Grid- und Tiefenparameter
grid_size = 14
patch_size = 16
pad_top_px = 48
pad_bottom_px = 48

max_depth_distance = 10.0  

os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                         (0.26862954, 0.26130258, 0.27577711))
])


def load_and_compute_ground_truth():
    """Berechnet die echte Ground Truth aus den Sensordaten pro Timestamp."""
    if not os.path.exists(csv_robot_path) or not os.path.exists(csv_timestamps_path):
        print(f"⚠️ Sensor CSV oder Timestamps CSV nicht gefunden!")
        return {}

    df_robot = pd.read_csv(csv_robot_path, sep=';')
    df_ts = pd.read_csv(csv_timestamps_path, sep=',')

    df_robot.columns = df_robot.columns.astype(str).str.strip()
    df_ts.columns = df_ts.columns.astype(str).str.strip()

    weights = {'alpha': 0.2, 'beta': 0.2, 'gamma': 0.2, 'delta': 0.2, 'epsilon': 0.2}

    raw_trav = (
        weights['alpha'] * df_robot['odom_vel'].values +
        weights['beta'] * df_robot['lin_acc_mean'].values +
        weights['gamma'] * df_robot['lin_acc_std_dev'].values +
        weights['delta'] * df_robot['smoothness_mean'].values +
        weights['epsilon'] * df_robot['slope'].values
    )
    
    scaler = MinMaxScaler()
    df_robot['trav_score'] = scaler.fit_transform(raw_trav.reshape(-1, 1)).flatten()

    df_robot['time'] = pd.to_numeric(df_robot['time'], errors='coerce').astype(np.float64)
    df_ts['timestamp_sec'] = pd.to_numeric(df_ts['timestamp_sec'], errors='coerce').astype(np.float64)

    df_robot = df_robot.dropna(subset=['time']).sort_values('time')
    df_ts = df_ts.dropna(subset=['timestamp_sec']).sort_values('timestamp_sec')

    merged_df = pd.merge_asof(
        df_ts, 
        df_robot[['time', 'trav_score']], 
        left_on='timestamp_sec',
        right_on='time', 
        direction='nearest'
    )

    gt_map = {}
    col_img_name = 'filename' if 'filename' in merged_df.columns else ('image_name' if 'image_name' in merged_df.columns else None)
    
    for _, row in merged_df.iterrows():
        img_name = str(row[col_img_name]) if col_img_name else str(row.name)
        clean_key = os.path.splitext(os.path.basename(img_name))[0]
        gt_map[clean_key] = row['trav_score']
        gt_map[img_name] = row['trav_score']

    return gt_map


def create_grid_masks(pad_top_px=48, pad_bottom_px=48):
    pad_mask_2d = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    row_start = pad_top_px // patch_size
    row_end = grid_size - (pad_bottom_px // patch_size)
    pad_mask_2d[row_start:row_end, :] = True
    return pad_mask_2d.flatten()


def extract_maskclip_patch_features(model, img_path, pad_mask_1d, device):
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    mask_2d = pad_mask_1d.view(grid_size, grid_size).to(device)
    pixel_mask = mask_2d.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    image_tensor = image_tensor * pixel_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        visual = model.visual
        x = visual.conv1(image_tensor.type(model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1) + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        x = visual.transformer(x).permute(1, 0, 2)

        patch_tokens = visual.ln_post(x[:, 1:, :])
        if visual.proj is not None:
            patch_features = patch_tokens @ visual.proj
        else:
            patch_features = patch_tokens
            
        patch_features /= patch_features.norm(dim=-1, keepdim=True)

    return patch_features.squeeze(0).cpu().numpy()


def get_patch_depths(depth_map, grid_size=14):
    """Skaliert die Tiefenkarte herunter, toleriert geringe Disparitätsdaten."""
    h, w = depth_map.shape
    patch_h, patch_w = h / grid_size, w / grid_size
    
    patch_depths = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    
    for r in range(grid_size):
        for c in range(grid_size):
            r_start, r_end = int(r * patch_h), int((r + 1) * patch_h)
            c_start, c_end = int(c * patch_w), int((c + 1) * patch_w)
            
            patch_roi = depth_map[r_start:r_end, c_start:c_end]
            valid_pixels = patch_roi[(~np.isnan(patch_roi)) & (patch_roi > 0)]
            
            # Mindestens 5% valide Pixel reichen aus, um dem Patch eine Tiefe zu geben
            if len(valid_pixels) > (patch_roi.size * 0.05):
                patch_depths[r, c] = np.nanmedian(valid_pixels)
                
    return patch_depths.flatten()


def testing_regressor():
    # Ground Truth laden
    gt_map = load_and_compute_ground_truth()
    all_preds = []
    all_targets = []

    print("Loading Regressor and Scaler...")
    with open(regressor_path, 'rb') as f:
        saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data["scaler"]

    padding_mask_path = os.path.join(masks_dir, "padding_masks_depth.pt")
    
    if os.path.exists(padding_mask_path):
        print(f"Loading Padding mask from {padding_mask_path}...")
        mask_data = torch.load(padding_mask_path, map_location="cpu")
        pad_mask = mask_data[0]["pad_mask"].numpy() if isinstance(mask_data, list) else mask_data.numpy()
        pad_mask = pad_mask.astype(bool).flatten()[:196]
    else:
        pad_mask = create_grid_masks(pad_top_px, pad_bottom_px).numpy()

    test_files = [f for f in sorted(os.listdir(test_image_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Starting testing for {len(test_files)} images...")

    pad_mask_tensor = torch.tensor(pad_mask, dtype=torch.bool)

    for idx, file_name in enumerate(tqdm(test_files)):
        img_path = os.path.join(test_image_dir, file_name)
        patch_feats = extract_maskclip_patch_features(maskclip_model, img_path, pad_mask_tensor, device)

        depth_file_name = os.path.splitext(file_name)[0] + ".npy"
        depth_path = os.path.join(depth_dir, depth_file_name)

        if os.path.exists(depth_path):
            depth_map = np.load(depth_path)
            patch_depths = get_patch_depths(depth_map, grid_size=grid_size)
            depth_mask = (~np.isnan(patch_depths)) & (patch_depths <= max_depth_distance)
            
            # Einmalige Diagnoseausgabe beim ersten Bild
            if idx == 0:
                print(f"\n[DEBUG Frame 0] Max gemessene Tiefe in .npy: {np.nanmax(patch_depths):.2f}m")
                print(f"[DEBUG Frame 0] Patches <= 10m: {np.sum(depth_mask)} / 196")
        else:
            depth_mask = np.zeros(196, dtype=bool)

        combined_valid_mask = pad_mask & depth_mask
        heatmap_flat = np.full(196, np.nan, dtype=np.float32)

        if np.any(combined_valid_mask):
            valid_patch_feats = patch_feats[combined_valid_mask]
            valid_feats_scaled = scaler.transform(valid_patch_feats)
            valid_predictions = model.predict_proba(valid_feats_scaled)[:, 1]
            heatmap_flat[combined_valid_mask] = valid_predictions

            # Ground Truth Matching
            clean_file_key = os.path.splitext(file_name)[0]
            gt_score = gt_map.get(clean_file_key, gt_map.get(file_name, None))
            if gt_score is not None:
                all_preds.extend(valid_predictions)
                all_targets.extend([gt_score] * len(valid_predictions))

        heatmap_2d = heatmap_flat.reshape(14, 14)

        orig_img = np.array(Image.open(img_path).convert("RGB"))
        img_height, img_width = orig_img.shape[:2]

        valid_mask_2d = combined_valid_mask.reshape(14, 14)
        mask_resized = cv2.resize(valid_mask_2d.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST).astype(bool)

        heatmap_smooth_raw = cv2.resize(np.nan_to_num(heatmap_2d, nan=0.0), (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        heatmap_smooth_filtered = ndimage.gaussian_filter(heatmap_smooth_raw, sigma=2.5)
        heatmap_smooth_filtered = np.clip(heatmap_smooth_filtered, 0.0, 1.0)
        heatmap_smooth_masked = np.ma.masked_where(~mask_resized, heatmap_smooth_filtered)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(orig_img)
        im = ax.imshow(heatmap_smooth_masked, cmap="RdYlGn", alpha=0.55, vmin=0, vmax=1)
        ax.set_title(f"Traversierbarkeit bis {max_depth_distance}m ({file_name})")
        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
        cbar.set_label('Traversierungs-Wahrscheinlichkeit')

        save_path = os.path.join(output_dir, f"heatmap_{file_name}")
        plt.savefig(save_path, bbox_inches='tight')
        
        fig.clf()
        plt.close('all')

    # Auswertung gegen Ground Truth
    if len(all_targets) > 0:
        threshold = 0.5
        y_pred = (np.array(all_preds) >= threshold).astype(int)
        y_true = (np.array(all_targets) >= threshold).astype(int)

        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        # ---------------------------------------------------------
        # CONFUSION MATRIX BERECHNEN (Layout: TP FN / FP TN)
        # ---------------------------------------------------------
        cm_standard = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm_standard.ravel() if cm_standard.size == 4 else (0, 0, 0, 0)

        # Custom Matrix-Anordnung:
        # [[TP, FN],
        #  [FP, TN]]
        custom_cm = np.array([[tp, fn], 
                              [fp, tn]])

        print("\n" + "="*45)
        print("📊 GROUND TRUTH EVALUATION")
        print("="*45)
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC:      {mcc:.4f}\n")

        print("🧩 CONFUSION MATRIX (Text) [ Layout: TP FN / FP TN ]:")
        print(f"   True Positives  (TP): {tp:10d}  |  False Negatives (FN): {fn:10d}")
        print(f"   False Positives (FP): {fp:10d}  |  True Negatives  (TN): {tn:10d}\n")

        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Non-Traversable", "Traversable"], zero_division=0))
        print("="*45)

        # ---------------------------------------------------------
        # CONFUSION MATRIX ALS BILD SPEICHERN
        # ---------------------------------------------------------
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=custom_cm, display_labels=["Traversable (1)", "Non-Traversable (0)"])
        disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d')
        
        ax_cm.set_title("Confusion Matrix (Layout: TP FN / FP TN)")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")

        cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_save_path, dpi=300)
        plt.close(fig_cm)
        print(f"🖼️ Confusion Matrix Plot gespeichert unter: {cm_save_path}")

    else:
        print("\n⚠️ Keine übereinstimmenden Ground-Truth-Daten gefunden.")

    print(f"\n✅ Fertig! Heatmaps wurden unter '{output_dir}' gespeichert.")

if __name__ == "__main__":
    testing_regressor()