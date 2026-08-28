# Preprocessing

The preprocessing scripts were reduced to small dataset-specific entry points plus shared utility modules.

## Main improvements

- duplicated resize-and-padding code is shared in `image_utils.py`
- only stereo pairs that exist on both sides are processed
- depth generation iterates over actual image pairs instead of a hard-coded frame count
- depth visualization can be disabled with `SAVE_VISUALIZATIONS = False`
- OTAS conversion to the 14x14 patch grid is shared in `otas_utils.py`
- RELLIS and RoboNav OTAS files use the same meaning: `True = valid ground/path patch`
- RoboNav rosbag extraction shares image decoding and camera-info handling
- rectification pairs images by filename instead of relying only on sorted list positions
- comments and configuration blocks are compact and in English

The numerical camera parameters and dataset-specific topics from the supplied scripts were preserved.
