from pathlib import Path
from rosbag_utils import extract_robonav_bag

BAG_PATH = Path("datasets/robonav/mattro_route1/mattro_01_2022-06-23-07-29-23.bag")
OUTPUT_DIR = Path("datasets/robonav/mattro_route1")
LEFT_IMAGE_TOPIC = "/hazard_front/zed_node_front/left_raw/image_raw_color"
RIGHT_IMAGE_TOPIC = "/hazard_front/zed_node_front/right_raw/image_raw_color"
LEFT_CAMERA_TOPIC = "/hazard_front/zed_node_front/left_raw/camera_info"
RIGHT_CAMERA_TOPIC = "/hazard_front/zed_node_front/right_raw/camera_info"


if __name__ == "__main__":
    extract_robonav_bag(
        BAG_PATH, OUTPUT_DIR, LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC,
        LEFT_CAMERA_TOPIC, RIGHT_CAMERA_TOPIC,
    )
