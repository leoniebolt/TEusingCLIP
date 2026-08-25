# sources:
# https://www.geeksforgeeks.org/python/interpolation-in-python/
# https://www.geeksforgeeks.org/machine-learning/save-and-load-machine-learning-models-in-python-with-scikit-learn/
# https://www.geeksforgeeks.org/machine-learning/interpolation-in-machine-learning/
# https://medium.com/@nagasameer/image-processing-using-a-gaussian-blur-with-scipy-543078f22d8d

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, confusion_matrix, ConfusionMatrixDisplay

device = "cuda" if torch.cuda.is_available() else "cpu"

maskclip_model, preprocess = clip.load("ViT-B/16", device=device)
maskclip_model.eval()

test_image_dir = "datasets/robonav/mattro_route1/images/test/images_left_224"
regressor_path = "datasets/robonav/mattro_route1/OTAS_masked/regressor_OTAS.sav"
output_dir = "datasets/robonav/mattro_route1/OTAS_masked/OTAS_heatmaps"

base_dir = "datasets/robonav/mattro_route1"
otas_masks_dir = os.path.join(base_dir, "OTAS_masked")

# Sensordaten & Timestamps Pfade für die Ground Truth
csv_robot_path = os.path.join(base_dir, "csv_files/mattro_kantine.csv")
csv_timestamps_path = os.path.join(base_dir, "csv_files/timestamps_left.csv")

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


def extract_maskclip_patch_features(model, img_path, device):
    """Lädt ein Bild, führt MaskCLIP aus und gibt Patch-Features [196, 512] zurück."""
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

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


def apply_gaussian_interpolation_14x14(heatmap_14x14, valid_mask_14x14, sigma=1.0):
    """
    Führt eine normierte Gaußsche Interpolation auf dem 14x14 Grid aus,
    sodass NaN-Pixel nicht in die gültigen Daten hineingemischt werden.
    """
    # Ersetze NaN durch 0 für die Faltung
    values_filled = np.nan_to_num(heatmap_14x14, nan=0.0)
    mask_float = valid_mask_14x14.astype(np.float32)

    # Faltung der Werte und der Maske
    blurred_values = ndimage.gaussian_filter(values_filled, sigma=sigma, mode='nearest')
    blurred_mask = ndimage.gaussian_filter(mask_float, sigma=sigma, mode='nearest')

    # Normalisierung (Vermeidung von Division durch Null)
    blurred_mask[blurred_mask == 0] = 1e-5
    smoothed_heatmap = blurred_values / blurred_mask

    # Wertebereich begrenzen und Maskierung wieder anwenden
    smoothed_heatmap = np.clip(smoothed_heatmap, 0.0, 1.0)
    smoothed_heatmap[~valid_mask_14x14] = np.nan

    return smoothed_heatmap


def testing_regressor():
    # Sensordaten Ground Truth initialisieren
    gt_map = load_and_compute_ground_truth()
    all_preds = []
    all_targets = []

    print(f"Loading Regressor and Scaler from {regressor_path}...")
    with open(regressor_path, 'rb') as f:
        saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data["scaler"]

    # 1. Statische Padding-Maske
    padding_mask_path = os.path.join(base_dir, "padding_masks_OTAS.pt")
    if os.path.exists(padding_mask_path):
        pad_data = torch.load(padding_mask_path, map_location="cpu")
        if isinstance(pad_data, list):
            pad_mask = pad_data[0]["pad_mask"].numpy().astype(bool)
        elif isinstance(pad_data, torch.Tensor):
            pad_mask = pad_data.numpy().astype(bool).flatten()[:196]
        else:
            pad_mask = np.array(pad_data).astype(bool).flatten()[:196]
    else:
        pad_mask_2d = np.zeros((14, 14), dtype=bool)
        pad_mask_2d[3:11, :] = True
        pad_mask = pad_mask_2d.flatten()

    # 2. OTAS Masken
    otas_mask_path = os.path.join(otas_masks_dir, "otas_masks_test.pt")
    if os.path.exists(otas_mask_path):
        otas_dataset = torch.load(otas_mask_path, map_location="cpu")
        otas_lookup = {
            item["frame_id"]: (item["otas_mask_1d"].numpy().astype(bool) 
                               if isinstance(item["otas_mask_1d"], torch.Tensor) 
                               else np.array(item["otas_mask_1d"]).astype(bool))
            for item in otas_dataset
        }
    else:
        otas_lookup = {}

    test_files = [f for f in sorted(os.listdir(test_image_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Starting inference for {len(test_files)} test images...")

    # Custom Colormap: Ungültige NaN-Pixel (Maskiert) sind 100% transparent
    current_cmap = plt.cm.get_cmap("RdYlGn").copy()
    current_cmap.set_bad(color='none', alpha=0.0)

    for idx, file_name in enumerate(test_files):
        img_path = os.path.join(test_image_dir, file_name)
        patch_feats = extract_maskclip_patch_features(maskclip_model, img_path, device)

        # Invertierung: OTAS Hindernis -> ~raw_otas = Valider Boden
        if file_name in otas_lookup:
            raw_otas = otas_lookup[file_name]
            otas_valid = ~raw_otas  
            combined_valid_mask = pad_mask & otas_valid
        else:
            combined_valid_mask = pad_mask

        # Grid (196 Patches) mit NaN initialisieren
        heatmap_flat = np.full(196, np.nan, dtype=np.float32)

        if np.any(combined_valid_mask):
            valid_patch_feats = patch_feats[combined_valid_mask]
            valid_feats_scaled = scaler.transform(valid_patch_feats)
            
            # Wahrscheinlichkeiten eintragen
            preds = model.predict_proba(valid_feats_scaled)[:, 1]
            heatmap_flat[combined_valid_mask] = preds

            # Ground-Truth Vergleich aufbauen
            clean_file_key = os.path.splitext(file_name)[0]
            gt_score = gt_map.get(clean_file_key, gt_map.get(file_name, None))
            if gt_score is not None:
                all_preds.extend(preds)
                all_targets.extend([gt_score] * len(preds))

        # 14x14 Grids
        heatmap_14x14 = heatmap_flat.reshape(14, 14)
        valid_mask_14x14 = combined_valid_mask.reshape(14, 14)

        # ---------------------------------------------------------
        # GAUSS'SCHE INTERPOLATION
        # ---------------------------------------------------------
        smoothed_14x14 = apply_gaussian_interpolation_14x14(heatmap_14x14, valid_mask_14x14, sigma=1.2)

        orig_img = np.array(Image.open(img_path).convert("RGB"))
        img_height, img_width = orig_img.shape[:2]

        # Hochskalieren mit kubischer Interpolation für flüssige Kanten
        # NaNs vor dem cv2.resize temporär mit Null ersetzen
        smoothed_filled = np.nan_to_num(smoothed_14x14, nan=0.0)
        heatmap_resized = cv2.resize(smoothed_filled, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        
        mask_resized = cv2.resize(valid_mask_14x14.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        heatmap_resized[~mask_resized] = np.nan

        # Visualisierung
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.imshow(orig_img)
        im = ax.imshow(
            heatmap_resized, 
            cmap=current_cmap, 
            alpha=0.6, 
            vmin=0.0, 
            vmax=1.0
        )

        ax.set_title(f"Gaussian Smoothed Heatmap - {file_name}")
        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.05)
        cbar.set_label('Traversierungs-Wahrscheinlichkeit')

        save_path = os.path.join(output_dir, f"heatmap_{file_name}")
        plt.savefig(save_path, bbox_inches='tight')
        
        fig.clf()
        plt.close('all')

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_files):
            print(f"Processed: {idx + 1} / {len(test_files)} images")

    # Auswertung gegen die Ground Truth
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

    print(f"\nDone! Heatmaps saved in {output_dir}")

testing_regressor()