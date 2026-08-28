from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
FUTURE_WINDOW = 2.0
TRAV_THRESHOLD = 0.5
DEPTH_THRESHOLD = 5.0
TRAIN_TEST_SIZE = 0.25
RANDOM_STATE = 29
ROI_ROWS = (6, 12)
ROI_COLS = (3, 11)

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    base_dir: Path
    image_dir: Path
    timestamp_csv: Path
    robot_csv: Optional[Path] = None
    imu_csv: Optional[Path] = None
    odom_csv: Optional[Path] = None
    pad_top: int = 48
    pad_bottom: int = 48
    depth_dir: Optional[Path] = None
    otas_mask_file: Optional[Path] = None

MATTRO_BASE = Path("datasets/robonav/mattro_route1")
SPOT_BASE = Path("datasets/robonav/spot_route1a")
RELLIS_BASE = Path("datasets/rellis_3d/00000_00")

DATASETS = {
    "mattro": DatasetConfig(
        name="mattro",
        base_dir=MATTRO_BASE,
        image_dir=MATTRO_BASE / "images/test/left_224",
        timestamp_csv=MATTRO_BASE / "csv_files/timestamps_left.csv",
        robot_csv=MATTRO_BASE / "csv_files/mattro_kantine.csv",
        depth_dir=MATTRO_BASE / "depth_masked/depth_information",
        otas_mask_file=MATTRO_BASE / "OTAS_masked/otas_masks_test.pt",
    ),
    "spot": DatasetConfig(
        name="spot",
        base_dir=SPOT_BASE,
        image_dir=SPOT_BASE / "images/test/left_224",
        timestamp_csv=SPOT_BASE / "csv_files/timestamps_left.csv",
        robot_csv=SPOT_BASE / "csv_files/spot_kantine.csv",
        depth_dir=SPOT_BASE / "depth_masked/depth_information",
        otas_mask_file=SPOT_BASE / "OTAS_masked/otas_masks_test.pt",
    ),
    "rellis": DatasetConfig(
        name="rellis",
        base_dir=RELLIS_BASE,
        image_dir=RELLIS_BASE / "images/test/left_224",
        timestamp_csv=RELLIS_BASE / "csv_files/timestamps_left.csv",
        imu_csv=RELLIS_BASE / "csv_files/imu_data.csv",
        odom_csv=RELLIS_BASE / "csv_files/odometry.csv",
        pad_top=32,
        pad_bottom=32,
        depth_dir=RELLIS_BASE / "depth_information",
        otas_mask_file=RELLIS_BASE / "otas_masks.pt",
    ),
}

TRAIN_BASE = MATTRO_BASE
TRAIN_IMAGE_DIR = TRAIN_BASE / "images/train/left_224"
TRAIN_TIMESTAMP_CSV = TRAIN_BASE / "csv_files/timestamps_left.csv"
TRAIN_ROBOT_CSV = TRAIN_BASE / "csv_files/mattro_kantine.csv"
TRAIN_DEPTH_DIR = TRAIN_BASE / "depth_masked/depth_information"
TRAIN_OTAS_MASK_FILE = TRAIN_BASE / "OTAS_masked/otas_masks_train.pt"

MODEL_PATHS = {
    "trav": TRAIN_BASE / "trav_masked/regressor_trav_masked.sav",
    "otas": TRAIN_BASE / "OTAS_masked/regressor_OTAS.sav",
    "depth": TRAIN_BASE / "depth_masked/regressor_depth.sav",
}

OUTPUT_DIRS = {
    ("mattro", "trav"): MATTRO_BASE / "trav_masked/heatmaps",
    ("mattro", "otas"): MATTRO_BASE / "OTAS_masked/heatmaps",
    ("mattro", "depth"): MATTRO_BASE / "depth_masked/heatmaps",
    ("spot", "trav"): SPOT_BASE / "heatmaps/trav_masked",
    ("spot", "otas"): SPOT_BASE / "heatmaps/OTAS_masked",
    ("spot", "depth"): SPOT_BASE / "heatmaps/depth_masked",
    ("rellis", "trav"): RELLIS_BASE / "heatmaps/trav_masked",
    ("rellis", "otas"): RELLIS_BASE / "heatmaps/OTAS_masked",
    ("rellis", "depth"): RELLIS_BASE / "heatmaps/depth_masked",
}

ROBONAV_PREPROCESSING = {
    "mattro": {
        "base_dir": MATTRO_BASE,
        "rectification_baseline": 0.10,
        "depth_fx": 530.3,
        "depth_baseline": 0.12,
    },
    "spot": {
        "base_dir": SPOT_BASE,
        "rectification_baseline": 0.10,
        "depth_fx": 530.3,
        "depth_baseline": 0.12,
    },
}
