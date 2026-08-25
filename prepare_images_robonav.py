# Resize and pad images for MaskCLIP input
#
# Sources:
#   https://pillow.readthedocs.io/
#   https://docs.opencv.org/
#   https://numpy.org/doc/

from PIL import Image
from pathlib import Path
import os

# mattro
#dataset_dir = Path("datasets/robonav/mattro_route1")

# spot
dataset_dir = Path("datasets/robonav/spot_route1a")

# general
sets = ["train", "test"]
target_size = 224


def resize_with_padding(image, target_size=224):
    width, height = image.size
    scale = min(
        target_size / width,
        target_size / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (0,0,0))

    x = (target_size - new_width) // 2
    y = (target_size - new_height) // 2

    canvas.paste(image, (x, y))

    return canvas

for split in sets:
    input_dir = (dataset_dir / "images" / split / "rectified_left")
    output_dir = (dataset_dir / "images" / split / "images_left_224")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))

    print(f"Processing {split}: {len(files)} images")

    for i, file in enumerate(files):
        image = Image.open(file).convert("RGB")
        image = resize_with_padding(image, target_size)
        image.save(output_dir / file.name)

        if i % 100 == 0:
            print(f"{i}/{len(files)}")

print("Finished.")