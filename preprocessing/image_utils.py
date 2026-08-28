from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def resize_with_padding(image, target_size=224):
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    new_size = (round(width * scale), round(height * scale))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset = ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def image_names(directory):
    directory = Path(directory)
    return {p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}


def paired_image_names(left_dir, right_dir):
    return sorted(image_names(left_dir) & image_names(right_dir))


def process_image_pairs(filenames, left_dir, right_dir, output_left, output_right, target_size=224):
    left_dir, right_dir = Path(left_dir), Path(right_dir)
    output_left, output_right = Path(output_left), Path(output_right)
    output_left.mkdir(parents=True, exist_ok=True)
    output_right.mkdir(parents=True, exist_ok=True)

    total = len(filenames)
    for i, filename in enumerate(filenames, 1):
        with Image.open(left_dir / filename) as image:
            resize_with_padding(image.convert("RGB"), target_size).save(output_left / filename)
        with Image.open(right_dir / filename) as image:
            resize_with_padding(image.convert("RGB"), target_size).save(output_right / filename)
        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total} image pairs")
