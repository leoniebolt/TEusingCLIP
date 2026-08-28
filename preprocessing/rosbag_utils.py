from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader


def decode_image(msg):
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if getattr(msg, "encoding", "") == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def camera_info_row(msg, timestamp):
    row = {"TIME": timestamp}
    row.update({f"K{i}": value for i, value in enumerate(msg.K)})
    row.update({f"D{i}": value for i, value in enumerate(msg.D)})
    row.update({f"R{i}": value for i, value in enumerate(msg.R)})
    row.update({f"P{i}": value for i, value in enumerate(msg.P)})
    return row


def extract_robonav_bag(bag_path, output_dir, image_topic_left, image_topic_right,
                         camera_left_topic, camera_right_topic):
    bag_path, output_dir = Path(bag_path), Path(output_dir)
    left_dir = output_dir / "images/original/left"
    right_dir = output_dir / "images/original/right"
    csv_dir = output_dir / "csv_files"
    for directory in (left_dir, right_dir, csv_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timestamps = {"left": [], "right": []}
    cameras = {"left": [], "right": []}
    counters = {"left": 0, "right": 0}
    topic_to_side = {image_topic_left: "left", image_topic_right: "right"}
    camera_to_side = {camera_left_topic: "left", camera_right_topic: "right"}
    target_topics = set(topic_to_side) | set(camera_to_side)

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic in target_topics]
        print(f"Relevant connections: {len(connections)}")
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            if conn.topic in topic_to_side:
                side = topic_to_side[conn.topic]
                filename = f"frame_{counters[side]:06d}.png"
                target_dir = left_dir if side == "left" else right_dir
                cv2.imwrite(str(target_dir / filename), decode_image(msg))
                timestamps[side].append({"filename": filename, "timestamp_sec": timestamp * 1e-9})
                counters[side] += 1
            elif conn.topic in camera_to_side:
                cameras[camera_to_side[conn.topic]].append(camera_info_row(msg, timestamp))

    pd.DataFrame(timestamps["left"]).to_csv(csv_dir / "timestamps_left.csv", index=False)
    pd.DataFrame(timestamps["right"]).to_csv(csv_dir / "timestamps_right.csv", index=False)
    pd.DataFrame(cameras["left"]).to_csv(csv_dir / "camera_left.csv", index=False)
    pd.DataFrame(cameras["right"]).to_csv(csv_dir / "camera_right.csv", index=False)
    print(f"Left images: {len(timestamps['left'])}")
    print(f"Right images: {len(timestamps['right'])}")
    print(f"Saved to: {output_dir}")
