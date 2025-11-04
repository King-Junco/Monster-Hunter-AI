"""
Convert all images in the dataset to PNG format
Handles: .webp, .png.webp, .jpg, .jpeg
Preserves transparency for PNGs
"""

import os
from pathlib import Path
from PIL import Image


def convert_to_png(input_path, output_path):
    """
    Convert an image file to PNG format, preserving transparency where possible.

    Args:
        input_path: Path to input image
        output_path: Path to save PNG image
    """
    try:
        img = Image.open(input_path).convert("RGBA")  # Always support transparency
        img.save(output_path, "PNG")
        return True
    except Exception as e:
        print(f"  ❌ Error converting {input_path}: {e}")
        return False


def convert_all_images(data_dir='MonsterHunterPics', backup=True):
    """
    Convert all images in data directory to PNG format

    Args:
        data_dir: Directory containing category folders with images
        backup: If True, keep original files with .backup extension
    """
    print("=" * 80)
    print("IMAGE FORMAT CONVERTER")
    print("=" * 80)
    print(f"\nScanning directory: {data_dir}")

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\n❌ Error: Directory '{data_dir}' not found!")
        return

    categories = [p for p in data_path.iterdir() if p.is_dir()]
    if not categories:
        print(f"\n❌ Error: No category folders found in '{data_dir}'")
        return

    print(f"Found {len(categories)} categories\n")

    total_files = 0
    converted_files = 0
    already_png = 0
    errors = 0

    convert_extensions = ['.webp', '.jpg', '.jpeg', '.png.webp']

    for category_path in categories:
        print(f"Processing: {category_path.name}")
        category_conversions = 0

        for file_path in category_path.iterdir():
            if not file_path.is_file():
                continue

            total_files += 1
            name_lower = file_path.name.lower().replace('.png.webp', '.webp')

            # Skip if already PNG
            if name_lower.endswith('.png'):
                already_png += 1
                continue

            # Determine if conversion needed
            if not any(name_lower.endswith(ext) for ext in convert_extensions):
                continue

            # New output name
            new_name = Path(name_lower).stem + ".png"
            output_path = category_path / new_name

            # Backup logic
            original_path = file_path
            backup_path = None
            if backup and original_path != output_path:
                backup_path = original_path.with_suffix(original_path.suffix + ".backup")
                if not backup_path.exists():
                    original_path.rename(backup_path)
                file_to_convert = backup_path
            else:
                file_to_convert = original_path

            # Convert
            if convert_to_png(file_to_convert, output_path):
                converted_files += 1
                category_conversions += 1

                # Cleanup
                if backup and backup_path and backup_path.exists():
                    backup_path.unlink()
                elif not backup and file_to_convert.exists() and file_to_convert != output_path:
                    file_to_convert.unlink()
            else:
                errors += 1

        if category_conversions > 0:
            print(f"  ✓ Converted {category_conversions} images")
        else:
            print(f"  - No conversions needed")

    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Total files scanned: {total_files}")
    print(f"Already PNG: {already_png}")
    print(f"Converted to PNG: {converted_files}")
    print(f"Errors: {errors}")
    print()

    if converted_files > 0:
        print("✓ All images converted to PNG format!")
        if backup:
            print("✓ Original files backed up with .backup extension")
            print("\nTo remove backups after verifying conversions:")
            print(f"  find {data_dir} -name '*.backup' -delete")
    else:
        print("✓ All images already in correct format!")


def verify_conversions(data_dir='MonsterHunterPics'):
    """
    Verify all images are valid PNGs
    """
    print("\n" + "=" * 80)
    print("VERIFYING CONVERSIONS")
    print("=" * 80)

    data_path = Path(data_dir)
    categories = [p for p in data_path.iterdir() if p.is_dir()]

    all_valid = True
    total_images = 0

    for category_path in categories:
        png_files = list(category_path.glob("*.png"))
        total_images += len(png_files)

        for file_path in png_files:
            try:
                img = Image.open(file_path)
                img.verify()
            except Exception as e:
                print(f"❌ Invalid image: {file_path}")
                print(f"   Error: {e}")
                all_valid = False

    print(f"\nTotal PNG images: {total_images}")
    if all_valid:
        print("✓ All images are valid PNGs!")
    else:
        print("⚠ Some images have issues (see above)")

    return all_valid


def remove_backups(data_dir='MonsterHunterPics'):
    """
    Remove all .backup files after verification
    """
    print("\n" + "=" * 80)
    print("REMOVING BACKUP FILES")
    print("=" * 80)

    data_path = Path(data_dir)
    backups = list(data_path.rglob("*.backup"))
    print(f"\nFound {len(backups)} backup files.")
    confirm = input("Type 'DELETE' to confirm removal: ")

    if confirm.strip().upper() != 'DELETE':
        print("Cancelled.")
        return

    for b in backups:
        try:
            b.unlink()
        except Exception as e:
            print(f"Error removing {b}: {e}")

    print(f"\n✓ Removed {len(backups)} backup files")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert all images to PNG format')
    parser.add_argument('--dir', type=str, default='MonsterHunterPics',
                        help='Directory containing image folders')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify converted images')
    parser.add_argument('--remove-backups', action='store_true',
                        help='Remove all .backup files')

    args = parser.parse_args()

    if args.remove_backups:
        remove_backups(args.dir)
    else:
        convert_all_images(args.dir, backup=not args.no_backup)

        if args.verify:
            verify_conversions(args.dir)

        print("\nNext steps:")
        print("  1. Verify images look correct")
        print("  2. If satisfied, remove backups:")
        print(f"     python convert_images_to_png.py --dir {args.dir} --remove-backups")
        print("  3. Organize data into data/raw/ folder")
        print("  4. Run preprocessing: python src/data_preprocessing.py")
