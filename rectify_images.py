# sources/helper codes:
#   https://docs.opencv.org/
#   http://docs.ros.org/en/api/sensor_msgs/html/msg/CameraInfo.html

import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# config
# --------------------------------------------------

# mattro
dataset_dir = Path("datasets/robonav/mattro_route1")
baseline = 0.10

# spot
#dataset_dir = Path("datasets/robonav/spot_route1a")
#baseline = 0.10

# DO NOT CHANGE
camera_left_file = dataset_dir / "csv_files/camera_left.csv"
camera_right_file = dataset_dir / "csv_files/camera_right.csv"

train_left_dir = (dataset_dir / "images/train/images_left")
train_right_dir = (dataset_dir / "images/train/images_right")

train_output_left = (dataset_dir / "images/train/rectified_left")
train_output_right = (dataset_dir / "images/train/rectified_right")

test_left_dir = (dataset_dir / "images/test/images_left")
test_right_dir = (dataset_dir / "images/test/images_right")

test_output_left = (dataset_dir / "images/test/rectified_left")
test_output_right = (dataset_dir / "images/test/rectified_right")

train_output_left.mkdir(parents=True, exist_ok=True)
train_output_right.mkdir(parents=True, exist_ok=True)
test_output_left.mkdir(parents=True, exist_ok=True)
test_output_right.mkdir(parents=True, exist_ok=True)

left_cam = pd.read_csv(camera_left_file).iloc[0]
right_cam = pd.read_csv(camera_right_file).iloc[0]


def get_K(cam):
    return np.array(
        [
            [
                cam["K0"],
                cam["K1"],
                cam["K2"]
            ],
            [
                cam["K3"],
                cam["K4"],
                cam["K5"]
            ],
            [
                cam["K6"],
                cam["K7"],
                cam["K8"]
            ]
        ],
        dtype=np.float64
    )


# --------------------------------------------------
# distortion coefficients
# --------------------------------------------------

def get_D(cam):
    return np.array(
        [
            cam["D0"],
            cam["D1"],
            cam["D2"],
            cam["D3"],
            cam["D4"]
        ],
        dtype=np.float64
    )


# --------------------------------------------------
# get camera matrices
# --------------------------------------------------
K_left = get_K(left_cam)
D_left = get_D(left_cam)

K_right = get_K(right_cam)
D_right = get_D(right_cam)

print("K left:", K_left)
print("K right:", K_right)


# --------------------------------------------------
# determine image size
# --------------------------------------------------
example_images = sorted(train_left_dir.glob("*.png"))

if len(example_images) == 0:
    # If train is empty, use test
    example_images = sorted(test_left_dir.glob("*.png"))

if len(example_images) == 0:
    raise Exception("No images found in train or test left directory!")

example = cv2.imread(str(example_images[0]))

if example is None:
    raise Exception(f"Could not read image: {example_images[0]}")

height, width = example.shape[:2]
image_size = (width, height)

print("Image size:", image_size)

# stereo camera geometry
# Factory aligned stereo cameras
R = np.eye(3, dtype=np.float64)

# Right camera relative to left
T = np.array([
        [-baseline],
        [0],
        [0]
    ],
    dtype=np.float64
)

# compute rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_left,
    D_left,
    K_right,
    D_right,
    image_size,
    R,
    T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

# create rectification maps
map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    K_left,
    D_left,
    R1,
    P1,
    image_size,
    cv2.CV_32FC1
)

map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    K_right,
    D_right,
    R2,
    P2,
    image_size,
    cv2.CV_32FC1
)

# function to rectify one split
def rectify_split(
    split_name,
    left_dir,
    right_dir,
    output_left,
    output_right
):
    print(f"Processing {split_name}")

    # get images
    left_images = sorted(left_dir.glob("*.png"))
    right_images = sorted(right_dir.glob("*.png"))

    print("Left images:", len(left_images))
    print("Right images:", len(right_images))

    # make sure both cameras contain
    # the same number of images
    if len(left_images) != len(right_images):
        raise Exception(f"Number of left and right images differs " f"in {split_name}!")

    print("Images:", len(left_images))

    # rectify images
    
    for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))

        if left is None:
            print(f"Could not read: {left_path}")
            continue

        if right is None:
            print(f"Could not read: {right_path}")
            continue


        # rectify left
        left_rect = cv2.remap(
            left,
            map_left_x,
            map_left_y,
            cv2.INTER_LINEAR
        )

        # rectify right
        right_rect = cv2.remap(
            right,
            map_right_x,
            map_right_y,
            cv2.INTER_LINEAR
        )

        # save left
        cv2.imwrite(str(output_left / left_path.name), left_rect)

        # save right
        cv2.imwrite(str(output_right / right_path.name), right_rect)


        # progress
        if idx % 100 == 0:
            print(f"Processed " f"{idx}/{len(left_images)}")

    print("Saved left:", output_left)
    print("Saved right:", output_right)

# rectify TRAIN
rectify_split(
    "TRAIN",
    train_left_dir,
    train_right_dir,
    train_output_left,
    train_output_right
)

# rectify TEST
rectify_split(
    "TEST",
    test_left_dir,
    test_right_dir,
    test_output_left,
    test_output_right
)

print("Rectification finished!")
print("Train:", train_output_left)
print("Test:", test_output_left)