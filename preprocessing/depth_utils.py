from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def draw_legend(image, max_depth_m=5.0):
    h, w = image.shape[:2]
    legend_width, bar_width, margin = 80, 25, 15
    canvas = np.zeros((h, w + legend_width, 3), dtype=np.uint8)
    canvas[:, :w] = image
    bar_height = h - 2 * margin
    gradient = np.linspace(255, 0, bar_height, dtype=np.uint8)[:, None]
    bar = cv2.applyColorMap(np.repeat(gradient, bar_width, axis=1), cv2.COLORMAP_TURBO)
    x = w + margin
    canvas[margin:margin + bar_height, x:x + bar_width] = bar
    cv2.rectangle(canvas, (x, margin), (x + bar_width, margin + bar_height), (255, 255, 255), 1)
    for i in range(5):
        fraction = i / 4
        y = int(margin + (1.0 - fraction) * bar_height)
        cv2.line(canvas, (x + bar_width, y), (x + bar_width + 4, y), (255, 255, 255), 1)
        cv2.putText(canvas, f"{fraction * max_depth_m:.1f}m", (x + bar_width + 5, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def create_stereo_matcher(channels=1):
    block_size = 7
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=block_size,
        P1=8 * channels * block_size**2,
        P2=32 * channels * block_size**2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )


def paired_names(left_dir, right_dir):
    left_dir, right_dir = Path(left_dir), Path(right_dir)
    left = {p.name for p in left_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    right = {p.name for p in right_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    return sorted(left & right)


def generate_depth_maps(left_dir, right_dir, depth_dir, fx, baseline, max_depth=5.0,
                        save_visualizations=True, channels=1):
    left_dir, right_dir, depth_dir = Path(left_dir), Path(right_dir), Path(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    image_dir = depth_dir / "depth_images"
    if save_visualizations:
        image_dir.mkdir(parents=True, exist_ok=True)

    names = paired_names(left_dir, right_dir)
    stereo = create_stereo_matcher(channels=channels)
    print(f"Found {len(names)} stereo pairs")

    for i, name in enumerate(names, 1):
        left = cv2.imread(str(left_dir / name), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_dir / name), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            print(f"Skipping {name}: image could not be loaded")
            continue

        disparity = stereo.compute(left, right).astype(np.float32) / 16.0
        disparity[disparity <= 0] = np.nan
        depth = (fx * baseline) / disparity
        np.save(depth_dir / f"{Path(name).stem}.npy", depth)

        if save_visualizations:
            visible = np.clip(np.nan_to_num(depth, nan=0.0), 0, max_depth)
            depth_8u = (visible / max_depth * 255).astype(np.uint8)
            colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
            colored[visible == 0] = 0
            cv2.imwrite(str(image_dir / name), draw_legend(colored, max_depth))

        if i % 100 == 0 or i == len(names):
            print(f"Processed {i}/{len(names)} stereo pairs")
