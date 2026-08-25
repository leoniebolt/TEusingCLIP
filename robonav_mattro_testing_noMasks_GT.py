# sources:
# https://www.geeksforgeeks.org/python/interpolation-in-python/
# https://www.geeksforgeeks.org/machine-learning/save-and-load-machine-learning-models-in-python-with-scikit-learn/

# input: model, ganzes unbekanntes Bild
# model auf unbekannten ganzen Bildern laufen lassen
# output: trav Vektor pro Bildpatch und heatmap
# Status: 100%

import os
import pickle
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib
# Extrem wichtig für Headless-Systeme / schnelle Schleifen-Plots:
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
maskclip_model.eval() # Modell in Evaluierungsmodus versetzen

# Pfade
test_image_dir = "datasets/robonav/mattro_route1/images/test/images_left_224"
regressor_path = "datasets/robonav/mattro_route1/naked/regressor_naked.sav"
output_dir = "datasets/robonav/mattro_route1/naked/heatmaps"
masks_dir = "datasets/robonav/mattro_route1/naked"

# Sensordaten & Timestamps Pfade
csv_robot_path = "datasets/robonav/mattro_route1/csv_files/mattro_kantine.csv"
csv_timestamps_path = "datasets/robonav/mattro_route1/csv_files/timestamps_left.csv"

os.makedirs(output_dir, exist_ok=True)

# Normalisierung für das Backbone (CLIP Standards)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                         (0.26862954, 0.26130258, 0.27577711))
])


def load_and_compute_ground_truth():
    """Lädt die Roboter-Sensordaten und berechnet den echten Traversability Score pro Timestamp."""
    print("Computing Sensor Ground Truth Trav Scores...")
    if not os.path.exists(csv_robot_path) or not os.path.exists(csv_timestamps_path):
        print(f"⚠️ Warning: Sensor CSV ({csv_robot_path}) or Timestamps CSV ({csv_timestamps_path}) not found!")
        return {}

    # CSVs laden
    df_robot = pd.read_csv(csv_robot_path, sep=';')
    df_ts = pd.read_csv(csv_timestamps_path, sep=',')

    # Spaltennamen säubern (eventuelle Leerzeichen/Tabs entfernen)
    df_robot.columns = df_robot.columns.astype(str).str.strip()
    df_ts.columns = df_ts.columns.astype(str).str.strip()

    # Gewichtung wie im Training
    weights = {
        'alpha': 0.2,
        'beta': 0.2,
        'gamma': 0.2,
        'delta': 0.2,
        'epsilon': 0.2
    }

    # Raw Trav Score berechnen
    raw_trav = (
        weights['alpha'] * df_robot['odom_vel'].values +
        weights['beta'] * df_robot['lin_acc_mean'].values +
        weights['gamma'] * df_robot['lin_acc_std_dev'].values +
        weights['delta'] * df_robot['smoothness_mean'].values +
        weights['epsilon'] * df_robot['slope'].values
    )
    
    scaler = MinMaxScaler()
    df_robot['trav_score'] = scaler.fit_transform(raw_trav.reshape(-1, 1)).flatten()

    # Datentypen der Zeitspalten zwingend und explizit als float64 erzwingen
    df_robot['time'] = pd.to_numeric(df_robot['time'], errors='coerce').astype(np.float64)
    df_ts['timestamp_sec'] = pd.to_numeric(df_ts['timestamp_sec'], errors='coerce').astype(np.float64)

    # Ungültige Timestamps entfernen (falls NAs entstanden sind)
    df_robot = df_robot.dropna(subset=['time'])
    df_ts = df_ts.dropna(subset=['timestamp_sec'])

    # Zeitspalten sortieren (Pflicht für merge_asof)
    df_robot = df_robot.sort_values('time')
    df_ts = df_ts.sort_values('timestamp_sec')

    # Merge über die jeweiligen Zeitspalten
    merged_df = pd.merge_asof(
        df_ts, 
        df_robot[['time', 'trav_score']], 
        left_on='timestamp_sec',
        right_on='time', 
        direction='nearest'
    )

    # Dictionary aufbauen: Key = Bildname / Clean Key -> Value = Sensor Trav Score
    gt_map = {}
    col_img_name = 'filename' if 'filename' in merged_df.columns else ('image_name' if 'image_name' in merged_df.columns else None)
    
    for _, row in merged_df.iterrows():
        img_name = str(row[col_img_name]) if col_img_name else str(row.name)
        clean_key = os.path.splitext(os.path.basename(img_name))[0]
        gt_map[clean_key] = row['trav_score']

    print(f"Successfully computed ground truth scores for {len(gt_map)} timestamps.")
    return gt_map


def extract_maskclip_patch_features(model, img_path, device):
    """Lädt ein Bild, führt MaskCLIP aus und gibt Patch-Features [196, 512] als NumPy-Array zurück."""
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


def evaluate_f1_and_mcc(all_predictions, all_ground_truths, threshold=0.5):
    """Rechnet F1-Score, MCC und Confusion Matrix auf Patch-Ebene aus."""
    y_pred = (np.array(all_predictions) >= threshold).astype(int)
    y_true = (np.array(all_ground_truths) >= threshold).astype(int)

    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Standard scikit-learn order: [[TN, FP], [FN, TP]]
    cm_standard = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm_standard.ravel() if cm_standard.size == 4 else (0, 0, 0, 0)

    # Anforderung:
    # [[TP, FN],
    #  [FP, TN]]
    custom_cm = np.array([[tp, fn], 
                          [fp, tn]])

    print("\n" + "="*45)
    print(f"📊 EVALUATION METRICS (Threshold = {threshold})")
    print("="*45)
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC:      {mcc:.4f}\n")

    print("🧩 CONFUSION MATRIX (Layout: TP FN / FP TN):")
    print(f"   True Positives  (TP): {tp:10d}  |  False Negatives (FN): {fn:10d}")
    print(f"   False Positives (FP): {fp:10d}  |  True Negatives  (TN): {tn:10d}\n")

    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Traversable", "Traversable"], zero_division=0))
    print("="*45)

    # Confusion Matrix als Plot speichern
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

    return f1, mcc


def testing_regressor():
    all_preds = []
    all_targets = []

    # 1. Ground Truth laden & berechnen
    gt_map = load_and_compute_ground_truth()

    # 2. Regressor & Scaler laden
    print("Loading Regressor and Scaler...")
    with open(regressor_path, 'rb') as f:
        saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data["scaler"]

    # ---------------------------------------------------------
    # GLOBALE MASKE LADEN & SAUBER ENTPACKEN
    # ---------------------------------------------------------
    padding_mask_path = os.path.join(masks_dir, "padding_masks.pt")
    
    if os.path.exists(padding_mask_path):
        print(f"Loading Padding-mask from {padding_mask_path}...")
        raw_mask_data = torch.load(padding_mask_path, map_location="cpu")
        
        # Falls als List-of-Dicts gespeichert ([{"pad_mask": ...}]):
        if isinstance(raw_mask_data, list) and len(raw_mask_data) > 0 and isinstance(raw_mask_data[0], dict):
            mask_tensor = raw_mask_data[0]["pad_mask"]
        else:
            mask_tensor = raw_mask_data

        if isinstance(mask_tensor, torch.Tensor):
            valid_mask = mask_tensor.numpy()
        else:
            valid_mask = np.array(mask_tensor)
            
        valid_mask = valid_mask.astype(bool).flatten()[:196]
        print(f"✅ Mask successfully loaded! Valid Patches: {np.sum(valid_mask)} / 196 (Padding Patches: {196 - np.sum(valid_mask)})")
    else:
        print("⚠️ Warning: No mask file found. Proceeding without masking.")
        valid_mask = np.ones(196, dtype=bool)

    # 3. Alle Test-Bilder laden
    test_files = [f for f in sorted(os.listdir(test_image_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Starting for {len(test_files)} test images from {test_image_dir}...")

    # Hauptschleife
    for idx, file_name in enumerate(test_files):
        img_path = os.path.join(test_image_dir, file_name)
        clean_file_key = os.path.splitext(os.path.basename(file_name))[0]

        # 1. MaskCLIP Features extrahieren -> Shape: [196, 512]
        patch_feats = extract_maskclip_patch_features(maskclip_model, img_path, device)

        # ---------------------------------------------------------
        # 2. BERECHNUNG: NUR AUF VALIDEN PATCHES AUSFÜHREN
        # ---------------------------------------------------------
        heatmap_flat = np.zeros(196, dtype=np.float32)

        if np.any(valid_mask):
            # Nur valide Features an den Regressor übergeben
            valid_patch_feats = patch_feats[valid_mask]
            valid_feats_scaled = scaler.transform(valid_patch_feats)
            valid_predictions = model.predict_proba(valid_feats_scaled)[:, 1]

            # Vorhersagen zurück ins Grid schreiben
            heatmap_flat[valid_mask] = valid_predictions

            # Sensordaten-GT vergleichen
            if clean_file_key in gt_map:
                frame_gt_score = gt_map[clean_file_key]
                num_valid_patches = np.sum(valid_mask)
                valid_gt_labels = np.full(num_valid_patches, frame_gt_score)

                all_preds.extend(valid_predictions)
                all_targets.extend(valid_gt_labels)

        heatmap_2d = heatmap_flat.reshape(14, 14)
        valid_mask_2d = valid_mask.reshape(14, 14)

        # ---------------------------------------------------------
        # 3. VISUALISIERUNG: HARD MASKING GEGEN PADDING
        # ---------------------------------------------------------
        orig_img = np.array(Image.open(img_path).convert("RGB"))
        img_height, img_width = orig_img.shape[:2]

        # Maske exakt auf volle Bildgröße skalieren
        valid_mask_resized = cv2.resize(valid_mask_2d.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST).astype(bool)

        # A) Blockig Heatmap (Ungültige Patches ausblenden)
        heatmap_blocky_raw = cv2.resize(heatmap_2d, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        heatmap_blocky_masked = np.ma.masked_where(~valid_mask_resized, heatmap_blocky_raw)

        # B) Smooth Heatmap (Normalisierter Gauß-Filter ohne Rand-Bleeding)
        smoothed_vals = ndimage.gaussian_filter(heatmap_2d, sigma=0.5)
        weight_mask = ndimage.gaussian_filter(valid_mask_2d.astype(float), sigma=0.5)
        weight_mask[weight_mask == 0] = 1.0 

        heatmap_smooth_base = smoothed_vals / weight_mask
        heatmap_smooth_raw = cv2.resize(heatmap_smooth_base, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        heatmap_smooth_raw = np.clip(heatmap_smooth_raw, 0.0, 1.0)
        
        # MASKIEREN: Ungültige Bereiche zu 100% transparent schalten
        heatmap_smooth_masked = np.ma.masked_where(~valid_mask_resized, heatmap_smooth_raw)

        # Plot erzeugen
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Blockig Plot
        axes[0].imshow(orig_img)
        axes[0].imshow(heatmap_blocky_masked, cmap="RdYlGn", alpha=0.5, vmin=0, vmax=1)
        axes[0].set_title(f"Blockig ({file_name})")
        axes[0].axis("off")

        # Flüssig Plot
        axes[1].imshow(orig_img)
        im = axes[1].imshow(heatmap_smooth_masked, cmap="RdYlGn", alpha=0.55, vmin=0, vmax=1)
        axes[1].set_title("Flüssige Heatmap")
        axes[1].axis("off")

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.046, pad=0.10)
        cbar.set_label('Traversierungs-Wahrscheinlichkeit')

        plt.subplots_adjust(bottom=0.15, top=0.9, wspace=0.1)
        save_path = os.path.join(output_dir, f"heatmap_{file_name}")
        plt.savefig(save_path, bbox_inches='tight')
        
        # RAM-Cleanup
        fig.clf()
        plt.close('all')

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_files):
            print(f"Processed: {idx + 1} / {len(test_files)} images")

    # ---------------------------------------------------------
    # 4. F1 SCORE, MCC UND CONFUSION MATRIX BERECHNEN UND AUSGEBEN
    # ---------------------------------------------------------
    if len(all_targets) > 0:
        evaluate_f1_and_mcc(all_preds, all_targets, threshold=0.5)
    else:
        print("\n⚠️ No Sensor Ground Truth matched to compute F1 Score & MCC.")

    print(f"\nDone! Heatmaps saved in {output_dir}")


testing_regressor()