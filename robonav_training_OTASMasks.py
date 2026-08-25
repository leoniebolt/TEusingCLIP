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
# https://github.com/SimonSchwaiger/otas/blob/main/demo.ipynb
 

from PIL import Image
import torch
import os
import csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from maskclip_onnx import clip
import argparse
import cv2
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# paths
csv_datafile_dir = "datasets/robonav/mattro_route1/csv_files/mattro_kantine.csv"
image_dir = "datasets/robonav/mattro_route1/images/train/images_left_224"
output_dir_features = "datasets/robonav/mattro_route1/OTAS_masked"
feature_file = os.path.join(output_dir_features, "maskclip_features_OTAS.pt")
padding_mask_file = os.path.join(output_dir_features, "padding_masks_OTAS.pt")
trav_mask_file = os.path.join(output_dir_features, "trav_masks_OTAS.pt")
otas_mask_file = os.path.join(output_dir_features, "OTAS_masks/otas_masks_train.pt") # NEU!

# trav parameters
trav_threshold = 0.5
future_window = 2.0

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

def extract_maskclip_patch_features(model, image_tensor):
    """Gibt sowohl das globale Feature [512] als auch die Patch-Features [196, 512] zurück."""
    with torch.no_grad():
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


def create_grid_masks(pad_top_px, pad_bottom_px, roi_row_range=(6, 12), roi_col_range=(4, 10)):
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

    #print(f"Padding Mask: {pad_mask_1d.sum().item()} / 196 Patches active.")
    #print(f"Traversability Mask:    {trav_mask_1d.sum().item()} / 196 Patches active.")

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

        global_feat, patch_feats = extract_maskclip_patch_features(model, image)
        future_trav_score = float(trav_scores[future_idx])

        feature_frame_data = {
            "frame_id": img_name,
            "frame_idx": idx,
            "future_idx": future_idx,
            "trav_score": future_trav_score,
            "trav_label": 1 if future_trav_score >= trav_threshold else 0,
            "global_feature": global_feat,
            "patch_features": patch_feats,
        }

        feature_dataset.append(feature_frame_data)
        padding_mask_dataset.append({"pad_mask": pad_mask_1d})
        trav_mask_dataset.append({"trav_mask": trav_mask_1d})

        if (idx + 1) % 100 == 0:
            print(f"Processed: {idx + 1} / {len(image_files)}")

    # save as .pt
    torch.save(feature_dataset, feature_file)
    torch.save(padding_mask_dataset, padding_mask_file)
    torch.save(trav_mask_dataset, trav_mask_file)

    print(f"\n✅ Dataset saved under: {output_dir_features}")


def train_logistic_regressor():
    print("Loading Dataset for training logistic regressor...")
    feature_dataset = torch.load(feature_file)
    padding_mask_dataset = torch.load(padding_mask_file)
    trav_mask_dataset = torch.load(trav_mask_file)
    
    # load OTAS-mask
    otas_dataset = torch.load(otas_mask_file)
    # Dictionary for fast access per frame-ID (from Gemini)
    otas_lookup = {
        item["frame_id"]: (item["otas_mask_1d"] if isinstance(item["otas_mask_1d"], torch.Tensor) else torch.tensor(item["otas_mask_1d"])).bool()
        for item in otas_dataset
    }

    X_list = []
    y_list = []

    for feat_sample, pad_sample, trav_sample in zip(feature_dataset, padding_mask_dataset, trav_mask_dataset):
        frame_id = feat_sample["frame_id"]
        patch_feats = feat_sample["patch_features"]  # [196, 512]
        label = feat_sample["trav_label"]            # 0 oder 1
        
        pad_mask = pad_sample["pad_mask"].bool()            # [196] Bool
        trav_mask = trav_sample["trav_mask"].bool()        # [196] Bool
        
        # Reliable retrieval with a fallback if the frame_id is missing (from Gemini)
        otas_mask = otas_lookup.get(frame_id, torch.ones(196, dtype=torch.bool))

        combined_valid_mask = pad_mask & trav_mask & otas_mask

        selected_patches = patch_feats[combined_valid_mask].numpy()

        if len(selected_patches) == 0:
            continue

        X_list.append(selected_patches)
        y_list.append(np.full(len(selected_patches), label))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"Preparing data: {X.shape[0]} Valid Patches total (after OTAS filtering), {X.shape[1]} feature-dimensions.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=29, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training logistic regressor...")
    regressor = LogisticRegression(max_iter=1000)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    y_prob = regressor.predict_proba(X_test)[:, 1]

    # save model and scaler
    regressor_name = 'regressor_OTAS.sav'
    save_full_path = os.path.join(output_dir_features, regressor_name)
    
    model_data = {
        "model": regressor,
        "scaler": scaler
    }
    
    with open(save_full_path, 'wb') as f:
        pickle.dump(model_data, f)

    print("\n" + "="*40)
    print("RESULTS WITH OTAS FILTERING")
    print("="*40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n✅ Model saved under: {save_full_path}")

process_and_save_dataset()
train_logistic_regressor()