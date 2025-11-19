from pathlib import Path
from PIL import Image
import shutil

def convert_to_png(input_path, backup_dir=None):
    """
    Convert a single image file to PNG format in place, preserving transparency.
    If backup_dir is provided, original files are moved there before conversion.
    """
    try:
        img = Image.open(input_path).convert("RGBA")  # Always support transparency
        output_path = input_path.with_suffix(".png")

        # Backup original if backup_dir is specified and file is not already PNG
        if backup_dir and input_path.suffix.lower() != ".png":
            # Preserve folder structure inside backup folder
            relative_path = input_path.relative_to(input_path.parents[2])  # raw/category/file.ext
            backup_path = backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(input_path), str(backup_path))
            file_to_save = backup_path
        else:
            file_to_save = input_path

        # Save the PNG
        img.save(output_path, "PNG")
        return True
    except Exception as e:
        print(f"  ❌ Error converting {input_path}: {e}")
        return False


def convert_all_in_raw(raw_dir='raw', backup=True):
    """
    Convert all images in 'raw' folder and its subfolders to PNG in place.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print(f"❌ Directory '{raw_dir}' not found!")
        return

    backup_dir = raw_path / "backup" if backup else None
    if backup_dir:
        backup_dir.mkdir(exist_ok=True)

    total_files = 0
    converted_files = 0
    already_png = 0

    print(f"\nScanning '{raw_dir}' recursively...\n")

    for file_path in raw_path.rglob("*"):  # recursive search
        if not file_path.is_file():
            continue

        total_files += 1

        if file_path.suffix.lower() == ".png":
            already_png += 1
            continue

        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".webp", ".png.webp"]:
            continue

        if convert_to_png(file_path, backup_dir=backup_dir):
            converted_files += 1

    print("\nConversion summary:")
    print(f"  Total files scanned: {total_files}")
    print(f"  Already PNG: {already_png}")
    print(f"  Converted to PNG: {converted_files}")
    if backup:
        print(f"  Original files backed up in: {backup_dir}")
    print(f"\n✓ All done! Images are now in PNG format in place.\n")


if __name__ == "__main__":
    convert_all_in_raw(raw_dir="data/raw", backup=True)
