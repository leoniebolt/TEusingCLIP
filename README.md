<div align="center">

# Self-Supervised Traversability Estimation using Vision-Language Model Features

### Master Thesis – Robotics Engineering

A framework for self-supervised visual traversability estimation using pretrained MaskCLIP features and robot telemetry.

</div>

---

## About

This repository contains the implementation developed for the master thesis **"Self-Supervised Traversability Estimation using Vision-Language Model Features"**.

The approach investigates whether visual representations learned by a pretrained Vision-Language Model (VLM) can be used for traversability estimation (TE) without task-specific fine-tuning of the visual backbone.

Robot telemetry is used to generate self-supervised traversability labels. RGB images are represented using patch-level features extracted from **MaskCLIP with a CLIP ViT-B/16 backbone**. A logistic regression classifier maps the resulting visual embeddings to traversability probabilities.

Three strategies for selecting spatially relevant image patches are evaluated:

* **Traversability mask (`trav`)** – fixed image region representing the area in front of the robot
* **OTAS mask (`otas`)** – semantic ground/path selection using OTAS
* **Depth mask (`depth`)** – geometric selection of patches within 5 m of the robot

The models are trained on data recorded with the **Mattro ROVO3** and evaluated in-domain as well as across different robotic platforms and datasets.

---

## Method Overview

The framework consists of an offline training stage and an evaluation stage.

### Offline Training

During training, robot telemetry is used to generate self-supervised traversability labels. MaskCLIP extracts patch-level visual features from the corresponding RGB images. Depending on the selected masking approach, only spatially relevant patches are used for training the logistic regression classifier.

### Testing

During testing, the trained classifier is applied to MaskCLIP patch features without retraining the visual backbone or classifier for the target platform.

The MaskCLIP visual encoder remains **frozen during all experiments**. Only the logistic regression classifier is trained.

---

## Datasets

The implementation supports three datasets / robotic platforms:

| Dataset / Platform                 | Role                                   | Usage                                   |
| ---------------------------------- | -------------------------------------- | --------------------------------------- |
| **RoboNav – Mattro ROVO3**         | Training + in-domain testing           | Model training and reference evaluation |
| **RoboNav – Boston Dynamics SPOT** | Cross-platform testing                 | Evaluation without retraining           |
| **RELLIS-3D – Warthog**            | Cross-dataset + cross-platform testing | Evaluation without retraining           |

The models trained on Mattro ROVO3 data are directly applied to SPOT and the Warthog platform from RELLIS-3D. This allows evaluating how well the learned traversability representation transfers without robot- or environment-specific retraining.

The datasets themselves are **not included in this repository**. They must be downloaded separately. Additional dataset-specific information and download links are provided in the corresponding dataset directories.

### Required RoboNav – Mattro Files

```text
datasets/robonav/mattro_route1/
├── mattro_01_2022-06-23-07-29-23.bag
└── csv_files/
    ├── 01_mattro_imu_220623_072923.csv
    └── mattro_kantine.csv
```

### Required RoboNav – SPOT Files

```text
datasets/robonav/spot_route1a/
├── spot_01_2022-06-22-10-12-24.bag
└── csv_files/
    └── spot_kantine.csv
```

### Required RELLIS-3D Files

```text
datasets/rellis_3d/00000_00/
└── 00000_00.bag
```

---

## Traversability Ground Truth

Traversability supervision is generated automatically from robot telemetry instead of manually annotated terrain labels.

For RoboNav, the traversability score is derived from robot locomotion information describing the interaction between the robot and the terrain. A future temporal window is used to associate terrain visible in an image with the physical interaction experienced by the robot shortly afterwards.

The resulting continuous traversability score is normalized to the range `[0, 1]` and converted into a binary training label using a threshold of `0.5`.

```text
score < 0.5  -> non-traversable
score >= 0.5 -> traversable
```

The default experimental parameters are:

| Parameter                |     Value |
| ------------------------ | --------: |
| Future window            |     2.0 s |
| Traversability threshold |       0.5 |
| Depth threshold          |     5.0 m |
| Input resolution         | 224 × 224 |
| MaskCLIP backbone        |  ViT-B/16 |
| Patch grid               |   14 × 14 |
| Number of patches        |       196 |
| Feature dimension        |       512 |

The telemetry-derived label represents the robot's experience along its driven trajectory. The selected masking strategies determine which visual patches receive this supervision during training.

---

## Spatial Patch Selection

Three masking strategies are implemented to select image regions that are relevant for traversability estimation.

### Traversability Mask

The `trav` configuration uses a predefined spatial region in the lower part of the image. This region approximates the terrain directly in front of the robot.

```text
Selected patches = fixed traversability region
```

This provides a simple spatial prior without requiring semantic segmentation or depth information.

### OTAS Mask

The `otas` configuration uses semantic terrain segmentation from **OTAS** to select ground/path regions.

```text
Selected patches = valid image region ∩ OTAS terrain mask
```

Patches outside the detected terrain regions are excluded from training and evaluation.

No additional fixed traversability ROI is applied.

### Depth Mask

The `depth` configuration uses stereo depth information to select terrain within a predefined distance from the robot.

```text
Selected patches = valid image region ∩ depth <= 5 m
```

Only patches corresponding to regions within the 5 m depth threshold are retained.

No additional fixed traversability ROI is applied.

---

## Project Structure

```text
.
├── project_config.py
├── train.py
├── test.py
│
├── preprocessing/
│   ├── extraction_rosbag_robonav.py
│   ├── extraction_spot_rosbag_robonav.py
│   ├── extraction_rosbag_rellis.py
│   ├── rectify_images.py
│   ├── prepare_images_robonav.py
│   ├── prepare_images_robonav_SPOT.py
│   ├── prepare_images_rellis.py
│   ├── robonav_save_depth.py
│   ├── rellis_save_depth.py
│   ├── robonav_extractOTAS.py
│   ├── rellis_extractOTASMask.py
│   ├── image_utils.py
│   ├── depth_utils.py
│   ├── otas_utils.py
│   └── rosbag_utils.py
│
├── utils/
│   ├── features.py
│   ├── masks.py
│   ├── depth.py
│   ├── traversability.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
│
├── Training.png
└── Testing.png
```

`project_config.py` contains shared paths and experimental parameters. Dataset-specific processing is kept separate where required, while common functionality is implemented in reusable utility modules.

---

## Requirements

The project is implemented in Python and primarily uses:

* Python
* PyTorch
* MaskCLIP / CLIP
* scikit-learn
* NumPy
* pandas
* OpenCV
* Pillow
* Matplotlib
* rosbags
* OTAS

### External Repositories

The implementation builds upon the following external projects:

**MaskCLIP ONNX**

https://github.com/RogerQi/maskclip_onnx

**OTAS**

https://github.com/SimonSchwaiger/otas

Because MaskCLIP and OTAS have different dependency requirements, **separate Conda environments are recommended**.

---

# Installation

## 1. Clone this Repository

```bash
git clone https://github.com/leoniebolt/TEusingCLIP.git
cd TEusingCLIP
```

---

## 2. MaskCLIP Environment

Create a separate Conda environment for the main preprocessing, training and evaluation pipeline.

For example:

```bash
conda create -n maskclip python=3.9
conda activate maskclip
```

Then install the required Python packages:

```bash
cd TEusingCLIP
pip install -r requirements.txt
```

Clone and install MaskCLIP_onnx according to its original installation instructions.

## 3. OTAS Environment

OTAS is executed in a separate environment due to its dependency requirements.

Create a new OTAS conda environment and clone and install OTAS.

# Running the Pipeline
---

## 1. Mattro Preprocessing

The Mattro ROVO3 dataset is used for model training and in-domain evaluation.

### Extract RoboNav Data

In the MaskCLIP environment:

```bash
python preprocessing/extraction_rosbag_robonav.py
```

### Rectify Stereo Images

```bash
python preprocessing/rectify_images.py --dataset mattro
```

### Prepare MaskCLIP Inputs

Create the randomized train/test image split and prepare the `224 × 224` inputs:

```bash
python preprocessing/prepare_images_robonav.py
```

### Generate Stereo Depth

```bash
python preprocessing/robonav_save_depth.py \
    --dataset mattro \
    --no-visualizations
```

### Generate OTAS Masks

Switch to the OTAS environment and generate masks for both subsets:

```bash
python preprocessing/robonav_extractOTAS.py \
    --dataset mattro \
    --split train

python preprocessing/robonav_extractOTAS.py \
    --dataset mattro \
    --split test
```

---

## 2. Training

All classifiers are trained exclusively using Mattro ROVO3 training data.

Switch back to the MaskCLIP environment.

### Fixed Traversability Mask

```bash
python train.py --mask trav
```

### OTAS Mask

```bash
python train.py --mask otas
```

### Depth Mask

```bash
python train.py --mask depth
```

For each configuration, MaskCLIP patch embeddings are extracted from the selected image regions. A `StandardScaler` is fitted to the training features, followed by a logistic regression classifier.

The MaskCLIP backbone itself is not trained or fine-tuned.

---

## 3. Mattro In-Domain Evaluation

Evaluate the three trained models on the held-out Mattro test data:

```bash
python test.py --dataset mattro --mask trav
python test.py --dataset mattro --mask otas
python test.py --dataset mattro --mask depth
```

To calculate the metrics without generating individual heatmaps:

```bash
python test.py --dataset mattro --mask trav --no-heatmaps
python test.py --dataset mattro --mask otas --no-heatmaps
python test.py --dataset mattro --mask depth --no-heatmaps
```

---

## 4. SPOT Preprocessing

SPOT is used to evaluate cross-platform transfer within the RoboNav dataset.

### Extract SPOT Data

In the MaskCLIP environment:

```bash
python preprocessing/extraction_spot_rosbag_robonav.py
```

### Rectify and Prepare Images

```bash
python preprocessing/rectify_images.py --dataset spot
python preprocessing/prepare_images_robonav_SPOT.py
```

### Generate Stereo Depth

```bash
python preprocessing/robonav_save_depth.py \
    --dataset spot \
    --no-visualizations
```

### Generate OTAS Masks

Switch to the OTAS environment:

```bash
python preprocessing/robonav_extractOTAS.py \
    --dataset spot \
    --split test
```

---

## 5. SPOT Cross-Platform Evaluation

No additional training or fine-tuning is performed on SPOT.

In the MaskCLIP environment:

```bash
python test.py --dataset spot --mask trav
python test.py --dataset spot --mask otas
python test.py --dataset spot --mask depth
```

The same classifiers trained on Mattro ROVO3 are used directly for SPOT.

---

## 6. RELLIS-3D Preprocessing

RELLIS-3D is used to evaluate transfer to a different dataset, environment and robotic platform.

### Extract and Prepare Data

In the MaskCLIP environment:

```bash
python preprocessing/extraction_rosbag_rellis.py
python preprocessing/prepare_images_rellis.py
```

### Generate Stereo Depth

```bash
python preprocessing/rellis_save_depth.py
```

### Generate OTAS Masks

Switch to the OTAS environment:

```bash
python preprocessing/rellis_extractOTASMask.py
```

---

## 7. RELLIS-3D Cross-Dataset Evaluation

No additional training or fine-tuning is performed on RELLIS-3D.

In the MaskCLIP environment:

```bash
python test.py --dataset rellis --mask trav
python test.py --dataset rellis --mask otas
python test.py --dataset rellis --mask depth
```

The classifiers trained on Mattro ROVO3 are directly applied to the RELLIS-3D data.

---

## Evaluation

The main evaluation metrics are:

* **F1 score**
* **Matthews Correlation Coefficient (MCC)**
* **Confusion matrix**

Additional metrics including accuracy, precision, recall and ROC-AUC are reported during classifier training.

The logistic regression classifier produces a traversability probability:

```text
MaskCLIP patch feature -> Logistic Regression -> P(traversable)
```

with

```text
P(traversable) ∈ [0, 1]
```

For binary evaluation, the predicted probability is converted using a threshold of `0.5`:

```text
P(traversable) < 0.5  -> non-traversable
P(traversable) >= 0.5 -> traversable
```

---

## Output

Depending on the selected configuration, the pipeline generates:

```text
trained logistic regression models
StandardScaler parameters
patch-level traversability probabilities
traversability heatmaps
confusion matrices
evaluation metrics
```

Heatmap generation can be disabled during evaluation using:

```bash
--no-heatmaps
```

This is useful when only numerical evaluation is required.

---

## Reproducibility

Important experimental parameters and dataset paths are defined centrally in:

```text
project_config.py
```

The default experimental configuration is:

```text
MaskCLIP backbone:       ViT-B/16
Image resolution:        224 × 224
Patch grid:              14 × 14
Number of patches:       196
Patch feature dimension: 512
Future window:           2.0 s
Traversability threshold: 0.5
Depth threshold:         5.0 m
```

A fixed random state is used where randomized data splitting is required to make the experiments reproducible.

---

## Results

The experiments investigate three main evaluation settings:

1. **In-domain evaluation:** Mattro ROVO3 → Mattro ROVO3
2. **Cross-platform evaluation:** Mattro ROVO3 → Boston Dynamics SPOT
3. **Cross-dataset and cross-platform evaluation:** Mattro ROVO3 → Warthog on RELLIS-3D

These experiments evaluate whether traversability information represented by pretrained MaskCLIP features can transfer between robotic platforms and environments without retraining the visual representation or classifier on the target platform.

Detailed quantitative and qualitative results are presented in the corresponding master thesis.

---

## Limitations

The telemetry-derived supervision describes the physical interaction of the robot with terrain along its driven trajectory. Assigning this signal to selected visual patches therefore provides self-supervised training labels, but does not represent dense pixel-wise traversability ground truth.

The visual backbone is kept frozen throughout the experiments. The experiments consequently evaluate the traversability information already available in the pretrained MaskCLIP representation rather than the performance achievable through task-specific fine-tuning.

Furthermore, the datasets provide different sensor configurations and telemetry information. Dataset-specific preprocessing is therefore required, particularly for the RELLIS-3D evaluation.

---

## Acknowledgements

This project builds upon the following open-source projects and datasets:

* [CLIP](https://github.com/openai/CLIP) – pretrained vision-language representation
* [MaskCLIP ONNX](https://github.com/RogerQi/maskclip_onnx) – dense CLIP feature extraction
* [OTAS](https://github.com/SimonSchwaiger/otas) – semantic terrain masking
* [RoboNav](https://github.com/ethz-asl/robonav) – robotic navigation dataset
* [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) – multimodal off-road dataset

Please refer to the original repositories and publications for their respective licenses and citation requirements.

---

## Citation

If you use this repository in academic work, please cite the corresponding master thesis:

```bibtex
@mastersthesis{bolt2026traversability,
  author = {Leonie Bolt},
  title  = {Self-Supervised Traversability Estimation using Vision-Language Model Features},
  school = {FH Technikum Wien},
  year   = {2026},
  type   = {Master's thesis}
}
```

---

## License

This repository contains code developed as part of a master's thesis and integrates or depends on external open-source projects.

Please check the licenses of **CLIP, MaskCLIP, OTAS, RoboNav and RELLIS-3D** before redistribution or reuse. The licenses of these external projects and datasets remain applicable to their respective components.
