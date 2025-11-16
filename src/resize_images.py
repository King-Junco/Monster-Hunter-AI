from pathlib import Path
from PIL import Image

def resize_and_pad(img, target_size=(256, 256)):
    """
    Resize image while preserving aspect ratio and pad with transparency to target size.
    """
    img = img.convert("RGBA")
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Create transparent background
    new_img = Image.new("RGBA", target_size, (0, 0, 0, 0))
    offset = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
    new_img.paste(img, offset)
    return new_img


def process_dataset(input_dir='D:\\Github-Desktop\\Monster-Hunter-AI\\data\\raw', output_dir='D:\\Github-Desktop\\Monster-Hunter-AI\\data\\processed', size=(256, 256)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    categories = [p for p in input_path.iterdir() if p.is_dir()]
    print(f"Resizing images from {input_dir} → {output_dir} ({size[0]}x{size[1]})")

    for category_path in categories:
        out_cat = output_path / category_path.name
        out_cat.mkdir(exist_ok=True)
        for img_path in category_path.glob("*.png"):
            try:
                img = Image.open(img_path)
                resized = resize_and_pad(img, size)
                resized.save(out_cat / img_path.name, "PNG")
            except Exception as e:
                print(f"Error resizing {img_path}: {e}")

    print("\n✓ All images resized and padded successfully!")


if __name__ == "__main__":
    process_dataset()
