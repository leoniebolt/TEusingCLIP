from pathlib import Path
from rosbag_utils import extract_robonav_bag

BAG_PATH = Path("datasets/robonav/spot_route1a/spot_01_2022-06-22-10-12-24.bag")
OUTPUT_DIR = Path("datasets/robonav/spot_route1a")
LEFT_IMAGE_TOPIC = "/hazard_front/zed_node_front/left_raw/image_raw_color"
RIGHT_IMAGE_TOPIC = "/hazard_front/zed_node_front/right_raw/image_raw_color"
LEFT_CAMERA_TOPIC = "/hazard_front/zed_node_front/left_raw/camera_info"
RIGHT_CAMERA_TOPIC = "/hazard_front/zed_node_front/right_raw/camera_info"


if __name__ == "__main__":
    extract_robonav_bag(
        BAG_PATH, OUTPUT_DIR, LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC,
        LEFT_CAMERA_TOPIC, RIGHT_CAMERA_TOPIC,
    )
