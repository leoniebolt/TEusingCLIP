# sources:
# https://github.com/openai/CLIP.git
# https://www.python-lernen.de/csv-datei-einlesen.htm
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# https://robonav.sai.tugraz.at/data/
# https://python4data.science/de/latest/clean-prep/scikit-learn-reprocessing.html
# https://www.geeksforgeeks.org/machine-learning/l1-l2-norms-in-sparse-modeling/
# https://www.geeksforgeeks.org/machine-learning/numpy-ndarray-flatten-function-python/
# https://docs.pytorch.org/docs/2.13/generated/torch.cat.html
# https://www.geeksforgeeks.org/python/python-opencv-cv2-rectangle-method/
# https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/
# https://www.geeksforgeeks.org/machine-learning/save-and-load-machine-learning-models-in-python-with-scikit-learn/
 
import os
import csv
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score
)

from maskclip_onnx import clip

# Setup Device & Modell
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# paths
csv_datafile_dir = "datasets/robonav/mattro_route1/csv_files/mattro_kantine.csv"
image_dir = "datasets/robonav/mattro_route1/images/train/images_left_224"
output_dir_features = "datasets/robonav/mattro_route1/depth"
feature_file = os.path.join(output_dir_features, "maskclip_features_depth.pt")
padding_mask_file = os.path.join(output_dir_features, "padding_masks_depth.pt")
trav_mask_file = os.path.join(output_dir_features, "trav_masks_depth.pt")
depth_dir = "datasets/robonav/mattro_route1/depth/depth_information"
left_dir = "datasets/robonav/mattro_route1/images/rectified/left"
right_dir = "datasets/robonav/mattro_route1/images/rectified/right"
regressor_file = "datasets/robonav/mattro_route1/depth/regressor_depth.sav"
depth_img_dir = os.path.join(depth_dir, "depth_images")

os.makedirs(depth_dir, exist_ok=True)
os.makedirs(depth_img_dir, exist_ok=True)
os.makedirs(output_dir_features, exist_ok=True)

# Parameters
trav_threshold = 0.5
future_window = 2.0
max_depth_distance = 5.0  

weights = {
    'alpha': 0.2,   # odom_vel
    'beta': 0.2,    # lin_acc_mean
    'gamma': 0.2,   # lin_acc_std_dev
    'delta': 0.2,   # smoothness_mean
    'epsilon': 0.2  # slope
}

# mask parameters
pad_top_px = 48
pad_bottom_px = 48
image_size = 224
patch_size = 16
grid_size = image_size // patch_size

# Camera parameters
fx = 530.3
baseline = 0.12


def create_grid_masks(pad_top_px, pad_bottom_px, roi_row_range=(6, 12), roi_col_range=(3, 11)):
    """Erstellt Boolean-Masken auf 14x14 Grid-Ebene."""
    pad_mask_2d = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    row_start = pad_top_px // patch_size
    row_end = grid_size - (pad_bottom_px // patch_size)
    pad_mask_2d[row_start:row_end, :] = True
    
    trav_mask_2d = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    r_start, r_end = roi_row_range
    c_start, c_end = roi_col_range
    
    r_start = max(row_start, r_start)
    r_end = min(row_end, r_end)
    
    trav_mask_2d[r_start:r_end, c_start:c_end] = True
    
    return trav_mask_2d.flatten(), pad_mask_2d.flatten()


def extract_maskclip_patch_features(model, image_tensor, pad_mask_1d):
    with torch.no_grad():
        mask_2d = pad_mask_1d.view(grid_size, grid_size).to(device)
        pixel_mask = mask_2d.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
        image_tensor = image_tensor * pixel_mask.unsqueeze(0).unsqueeze(0)

        global_feature = model.encode_image(image_tensor)
        global_feature /= global_feature.norm(dim=-1, keepdim=True)

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

    return global_feature.squeeze(0).cpu(), patch_features.squeeze(0).cpu()

def pad_and_resize_depth(depth_map, pad_top=48, pad_bottom=48, target_size=224):
    """Pads and resizes depth maps to match the exact letterboxing of training images."""
    inner_h = target_size - (pad_top + pad_bottom)
    
    # Resize depth map proportionally to fit the inner valid image height
    resized_depth = cv2.resize(depth_map, (target_size, inner_h), interpolation=cv2.INTER_NEAREST)
    
    # Create 224x224 canvas initialized with NaNs (so padding patches are ignored)
    padded_depth = np.full((target_size, target_size), np.nan, dtype=np.float32)
    
    # Place valid depth inside the padded canvas
    padded_depth[pad_top : pad_top + inner_h, :] = resized_depth
    
    return padded_depth

def get_patch_depths(depth_map, grid_size=14):
    """Calculates 14x14 patch medians on an ALREADY padded 224x224 depth map."""
    
    h, w = depth_map.shape
    patch_h, patch_w = h // grid_size, w // grid_size
    
    patch_depths = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    
    for r in range(grid_size):
        for c in range(grid_size):
            patch_roi = depth_map[r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w]
            valid_pixels = patch_roi[~np.isnan(patch_roi)]
            if len(valid_pixels) > 0:
                patch_depths[r, c] = np.nanmedian(valid_pixels)
                
    return patch_depths.flatten()


def process_and_save_dataset():
    df = pd.read_csv(csv_datafile_dir, sep=';')
    print(f"Number of datasets in csv: {len(df)}")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Images found: {len(image_files)}")

    shift_frames = int(future_window)

    trav_mask_1d, pad_mask_1d = create_grid_masks(
        pad_top_px=pad_top_px, 
        pad_bottom_px=pad_bottom_px,
        roi_row_range=(6, 12),
        roi_col_range=(3, 11)
    )

    raw_trav = (
        weights['alpha'] * df['odom_vel'].values +
        weights['beta'] * df['lin_acc_mean'].values +
        weights['gamma'] * df['lin_acc_std_dev'].values +
        weights['delta'] * df['smoothness_mean'].values +
        weights['epsilon'] * df['slope'].values
    )
    scaler = MinMaxScaler()
    trav_scores = scaler.fit_transform(raw_trav.reshape(-1, 1)).flatten()

    feature_dataset = []
    padding_mask_dataset = []
    trav_mask_dataset = []

    for idx, img_name in enumerate(image_files):
        future_idx = idx + shift_frames
        
        if future_idx >= len(trav_scores):
            print(f"Reached end at frame {idx}. No future data for +{future_window}s.")
            break

        image_path = os.path.join(image_dir, img_name)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        global_feat, patch_feats = extract_maskclip_patch_features(model, image, pad_mask_1d)
        future_trav_score = float(trav_scores[future_idx])

        depth_file_name = os.path.splitext(img_name)[0] + ".npy"
        depth_path = os.path.join(depth_dir, depth_file_name)
        
        if os.path.exists(depth_path):
            depth_map = np.load(depth_path)
            
            # 1. Align depth map to match training image padding
            aligned_depth = pad_and_resize_depth(
                depth_map, 
                pad_top=pad_top_px, 
                pad_bottom=pad_bottom_px, 
                target_size=image_size
            )
            
            # 2. Compute patch medians
            patch_depths = get_patch_depths(aligned_depth, grid_size=grid_size)
        else:
            patch_depths = np.full((grid_size * grid_size,), np.nan, dtype=np.float32)

        feature_frame_data = {
            "frame_id": img_name,
            "frame_idx": idx,
            "future_idx": future_idx,
            "trav_score": future_trav_score,
            "trav_label": 1 if future_trav_score >= trav_threshold else 0,
            "global_feature": global_feat,
            "patch_features": patch_feats,
            "patch_depths": torch.tensor(patch_depths, dtype=torch.float32)
        }

        feature_dataset.append(feature_frame_data)
        padding_mask_dataset.append({"pad_mask": pad_mask_1d})
        trav_mask_dataset.append({"trav_mask": trav_mask_1d})

        if (idx + 1) % 100 == 0:
            print(f"Processed: {idx + 1} / {len(image_files)}")

    torch.save(feature_dataset, feature_file)
    torch.save(padding_mask_dataset, padding_mask_file)
    torch.save(trav_mask_dataset, trav_mask_file)

    print(f"\n✅ Dataset saved under: {output_dir_features}")


def train_logistic_regressor():
    print("Loading Dataset for training regressor...")
    feature_dataset = torch.load(feature_file)
    padding_mask_dataset = torch.load(padding_mask_file)
    trav_mask_dataset = torch.load(trav_mask_file)

    X_list = []
    y_list = []

    for feat_sample, pad_sample, trav_sample in zip(feature_dataset, padding_mask_dataset, trav_mask_dataset):
        patch_feats = feat_sample["patch_features"]         # [196, 512]
        label = feat_sample["trav_label"]                   # 0 oder 1
        trav_mask = trav_sample["trav_mask"].numpy()        # [196] Bool
        pad_mask = pad_sample["pad_mask"].numpy()            # [196] Bool
        patch_depths = feat_sample["patch_depths"].numpy() # [196] Float

        # Filter criteria:
        # 1. Not within the padding region (pad_mask == True)
        # 2. Within the traversability ROI (trav_mask == True)
        # 3. Valid depth AND depth <= 5.0 m
        depth_mask = (~np.isnan(patch_depths)) & (patch_depths <= max_depth_distance)
        
        valid_mask = trav_mask & pad_mask & depth_mask

        selected_patches = patch_feats[valid_mask].numpy()

        if len(selected_patches) > 0:
            X_list.append(selected_patches)
            y_list.append(np.full(len(selected_patches), label))

    if len(X_list) == 0:
        print("❌ Keine Patches gefunden, die den Filterkriterien entsprechen!")
        return

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"Training Data shape: {X.shape[0]} Patches, {X.shape[1]} Feature-Dimensionen.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=29, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training Regressor...")
    regressor = LogisticRegression(max_iter=1000)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    y_prob = regressor.predict_proba(X_test)[:, 1]

    model_data = {
        "model": regressor,
        "scaler": scaler
    }
    
    with open(regressor_file, 'wb') as f:
        pickle.dump(model_data, f)

    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n✅ Model saved under: {regressor_file}")


if __name__ == "__main__":
    process_and_save_dataset()
    train_logistic_regressor()