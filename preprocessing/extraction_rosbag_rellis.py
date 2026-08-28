from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbag_utils import decode_image

BAG_PATH = Path("datasets/rellis_3d/00000_00/00000_00.bag")
OUTPUT_DIR = Path("datasets/rellis_3d/00000_00")
LEFT_TOPIC = "/nerian_stereo/left_image"
RIGHT_TOPIC = "/nerian_stereo/right_image"
IMU_TOPIC = "/os1_cloud_node/imu"
ODOM_TOPIC = "/odometry/filtered"


def main():
    left_dir = OUTPUT_DIR / "images/original/images_left"
    right_dir = OUTPUT_DIR / "images/original/images_right"
    csv_dir = OUTPUT_DIR / "csv_files"
    for directory in (left_dir, right_dir, csv_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timestamps_left, timestamps_right, imu_data, odom_data = [], [], [], []
    left_count = right_count = 0
    topics = {LEFT_TOPIC, RIGHT_TOPIC, IMU_TOPIC, ODOM_TOPIC}

    with AnyReader([BAG_PATH]) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            time_sec = timestamp * 1e-9

            if conn.topic == LEFT_TOPIC:
                filename = f"frame_{left_count:06d}.png"
                cv2.imwrite(str(left_dir / filename), decode_image(msg))
                timestamps_left.append({"filename": filename, "timestamp_nsec": timestamp, "timestamp_sec": time_sec})
                left_count += 1
            elif conn.topic == RIGHT_TOPIC:
                filename = f"frame_{right_count:06d}.png"
                cv2.imwrite(str(right_dir / filename), decode_image(msg))
                timestamps_right.append({"filename": filename, "timestamp_nsec": timestamp, "timestamp_sec": time_sec})
                right_count += 1
            elif conn.topic == IMU_TOPIC:
                imu_data.append({
                    "timestamp_nsec": timestamp, "timestamp_sec": time_sec,
                    "angular_velocity_x": msg.angular_velocity.x,
                    "angular_velocity_y": msg.angular_velocity.y,
                    "angular_velocity_z": msg.angular_velocity.z,
                    "linear_acceleration_x": msg.linear_acceleration.x,
                    "linear_acceleration_y": msg.linear_acceleration.y,
                    "linear_acceleration_z": msg.linear_acceleration.z,
                })
            elif conn.topic == ODOM_TOPIC:
                velocity = msg.twist.twist.linear
                odom_data.append({
                    "timestamp_nsec": timestamp, "timestamp_sec": time_sec,
                    "position_x": msg.pose.pose.position.x,
                    "position_y": msg.pose.pose.position.y,
                    "position_z": msg.pose.pose.position.z,
                    "orientation_x": msg.pose.pose.orientation.x,
                    "orientation_y": msg.pose.pose.orientation.y,
                    "orientation_z": msg.pose.pose.orientation.z,
                    "orientation_w": msg.pose.pose.orientation.w,
                    "linear_velocity_x": velocity.x,
                    "linear_velocity_y": velocity.y,
                    "linear_velocity_z": velocity.z,
                    "velocity": np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
                })

    pd.DataFrame(timestamps_left).to_csv(csv_dir / "timestamps_left.csv", index=False)
    pd.DataFrame(timestamps_right).to_csv(csv_dir / "timestamps_right.csv", index=False)
    pd.DataFrame(imu_data).to_csv(csv_dir / "imu_data.csv", index=False)
    pd.DataFrame(odom_data).to_csv(csv_dir / "odometry.csv", index=False)
    print(f"Left images: {len(timestamps_left)} | right images: {len(timestamps_right)}")
    print(f"IMU samples: {len(imu_data)} | odometry samples: {len(odom_data)}")


if __name__ == "__main__":
    main()
