# sources:

# input: rosbag
# Bilder mit Zeitstempel extrahieren (linke und rechte Kamera)
# Hälfte der Bilder in train, Hälfte in test
# output: csv mit Bild + Zeitstempel
# Status: 50%

from rosbags.highlevel import AnyReader
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

# config
bag_path = Path("datasets/robonav/mattro_route1/mattro_01_2022-06-23-07-29-23.bag")
output_dir = Path("datasets/robonav/mattro_route1")

image_dir_left = output_dir / "images/original/images_left"
image_dir_left.mkdir(parents=True, exist_ok=True)

image_dir_right = output_dir / "images/original/images_right"
image_dir_right.mkdir(parents=True, exist_ok=True)

image_dir_train = output_dir / "images/train"
image_dir_test = output_dir / "images/test"

image_topic_left = ("/hazard_front/zed_node_front/" "left_raw/image_raw_color")
image_topic_right = ("/hazard_front/zed_node_front/" "right_raw/image_raw_color")

camera_left_topic = ("/hazard_front/zed_node_front/" "left_raw/camera_info")
camera_right_topic = ("/hazard_front/zed_node_front/" "right_raw/camera_info")

camera_left = []
camera_right = []

timestamps_left = []
timestamps_right = []

image_counter_left = 0
image_counter_right = 0

# read rosbag
with AnyReader([bag_path]) as reader:
    for conn, timestamp, rawdata in reader.messages():
        msg = reader.deserialize(
            rawdata,
            conn.msgtype
        )
        # ROS timestamp in nanoseconds
        timestamp_sec = timestamp * 1e-9

# left image
        if conn.topic == image_topic_left:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, -1)

            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            filename = (f"frame_{image_counter_left:06d}.png")
            filepath = (image_dir_left / filename)

            cv2.imwrite(str(filepath), img)

            timestamps_left.append({
                    "filename": filename,
                    "timestamp_sec": timestamp_sec
                }
            )

            image_counter_left += 1


# right image
        elif conn.topic == image_topic_right:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, -1)

            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            filename = (f"frame_{image_counter_right:06d}.png")
            filepath = (image_dir_right / filename)

            cv2.imwrite(str(filepath), img)

            timestamps_right.append({
                    "filename": filename,
                    "timestamp_sec": timestamp_sec
                }
            )

            image_counter_right += 1

# camera info left
        elif conn.topic == camera_left_topic:
            camera_left.append({
                "TIME": timestamp,

                "K0": msg.K[0],
                "K1": msg.K[1],
                "K2": msg.K[2],
                "K3": msg.K[3],
                "K4": msg.K[4],
                "K5": msg.K[5],
                "K6": msg.K[6],
                "K7": msg.K[7],
                "K8": msg.K[8],
            
                "D0": msg.D[0],
                "D1": msg.D[1],
                "D2": msg.D[2],
                "D3": msg.D[3],
                "D4": msg.D[4],
            
                "R0": msg.R[0],
                "R1": msg.R[1],
                "R2": msg.R[2],
                "R3": msg.R[3],
                "R4": msg.R[4],
                "R5": msg.R[5],
                "R6": msg.R[6],
                "R7": msg.R[7],
                "R8": msg.R[8],
            
                "P0": msg.P[0],
                "P1": msg.P[1],
                "P2": msg.P[2],
                "P3": msg.P[3],
                "P4": msg.P[4],
                "P5": msg.P[5],
                "P6": msg.P[6],
                "P7": msg.P[7],
                "P8": msg.P[8],
                "P9": msg.P[9],
                "P10": msg.P[10],
                "P11": msg.P[11],
            }
            )

# camera info right
        elif conn.topic == camera_right_topic:
            camera_right.append({
                "TIME": timestamp,      

                "K0": msg.K[0],
                "K1": msg.K[1],
                "K2": msg.K[2],
                "K3": msg.K[3],
                "K4": msg.K[4],
                "K5": msg.K[5],
                "K6": msg.K[6],
                "K7": msg.K[7],
                "K8": msg.K[8],     
                "D0": msg.D[0],
                "D1": msg.D[1],
                "D2": msg.D[2],
                "D3": msg.D[3],
                "D4": msg.D[4],

                "R0": msg.R[0],
                "R1": msg.R[1],
                "R2": msg.R[2],
                "R3": msg.R[3],
                "R4": msg.R[4],
                "R5": msg.R[5],
                "R6": msg.R[6],
                "R7": msg.R[7],
                "R8": msg.R[8],

                "P0": msg.P[0],
                "P1": msg.P[1],
                "P2": msg.P[2],
                "P3": msg.P[3],
                "P4": msg.P[4],
                "P5": msg.P[5],
                "P6": msg.P[6],
                "P7": msg.P[7],
                "P8": msg.P[8],
                "P9": msg.P[9],
                "P10": msg.P[10],
                "P11": msg.P[11],
            }
           )

# save csv files
camera_left_df = pd.DataFrame(camera_left)
camera_right_df = pd.DataFrame(camera_right)

left_df = pd.DataFrame(timestamps_left)
right_df = pd.DataFrame(timestamps_right)

camera_left_df.to_csv(output_dir / "camera_left.csv", index=False)
camera_right_df.to_csv(output_dir / "camera_right.csv", index=False)

left_df.to_csv(output_dir / "timestamps_left.csv", index=False)
right_df.to_csv(output_dir / "timestamps_right.csv", index=False)

# summary
print("Left images:", len(camera_left_df))
print("Right images:", len(camera_right_df))
print("Saved:", output_dir)