import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


# -------------------------------------------------------
# Manifest Helpers
# -------------------------------------------------------
def load_manifest(manifest_path):
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[WARN] Manifest corrupted — starting fresh.")
    return {}


def save_manifest(manifest_path, manifest):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)


# -------------------------------------------------------
# String Helper
# -------------------------------------------------------
def sanitize_name(name):
    safe = re.sub(r'[<>:"/\\|?*]', "", name)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"[()]", "", safe)
    return safe.strip("_")


# -------------------------------------------------------
# Category Scraper
# -------------------------------------------------------
def get_monsters_by_category(category_url):
    """Scrape monster names from a Monster Hunter Wiki category or 'Wyvern type' article."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MonsterScraper/2.5)"}
    monsters = set()

    try:
        response = requests.get(category_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Method 1: Check category pages
        category_members = soup.find_all("div", class_="category-page__member-link")
        for member in category_members:
            link = member.find("a")
            if link and not any(x in link.text.lower() for x in ["category:", "template:", "user:", "file:"]):
                monsters.add(link.text.strip())

        # Method 2: Check tables and lists
        monster_tables = soup.find_all(["table", "ul"])
        for table in monster_tables:
            links = table.find_all("a")
            for link in links:
                if link.text and not any(x in link.get("href", "").lower() for x in ["category:", "template:", "user:", "file:"]):
                    monsters.add(link.text.strip())

        # Method 3: Check specific sections
        for section in soup.find_all(["div", "section"]):
            if "monster" in section.get("class", []):
                links = section.find_all("a")
                for link in links:
                    if link.text:
                        monsters.add(link.text.strip())

        print(f"Found {len(monsters)} monsters in category: {category_url}")
    except Exception as e:
        print(f"Error scraping category {category_url}: {e}")

    return list(monsters)


# -------------------------------------------------------
# Image Saver Helper
# -------------------------------------------------------
def save_images(imgs, output_folder, monster_name, base_url, headers):
    count = 0
    saved_urls = set()  # Avoid duplicates
    
    for img in imgs:
        img_url = img.get("src", "").split('?')[0]  # Remove URL parameters
        if not img_url or img_url in saved_urls:
            continue

        # Convert thumbnail URLs to full-size images
        img_url = img_url.replace('/thumb/', '/').split('/revision/')[0]
        if '/scale-to-width-down/' in img_url:
            img_url = img_url.split('/scale-to-width-down/')[0]

        img_url = urljoin(base_url, img_url)
        ext = os.path.splitext(urlparse(img_url).path)[-1].lower()
        
        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            continue

        filename = f"{sanitize_name(monster_name)}_{count}{ext}"
        filepath = os.path.join(output_folder, filename)

        try:
            img_data = requests.get(img_url, headers=headers, timeout=15).content
            if len(img_data) < 5000:  # Skip very small files
                continue
                
            with open(filepath, "wb") as f:
                f.write(img_data)
            saved_urls.add(img_url)
            count += 1
            print(f"  -> Saved: {filename}")
        except Exception as e:
            print(f"[FAIL] {img_url} — {e}")
            
    print(f"  -> {monster_name}: {count} images saved")
    return count


# -------------------------------------------------------
# Monster Gallery Scraper
# -------------------------------------------------------
def download_gallery_images(monster_name, output_folder):
    base_url = "https://monsterhunter.fandom.com/wiki/"
    page_url = f"{base_url}{monster_name.replace(' ', '_')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"\nProcessing {monster_name}...")
    print(f"URL: {page_url}")

    try:
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Could not load {monster_name}: {e}")
        return 0

    soup = BeautifulSoup(response.text, "html.parser")
    os.makedirs(output_folder, exist_ok=True)

    # Collect images from multiple sources
    imgs = []
    
    # 1. Check gallery pages
    gallery_urls = []
    for a in soup.find_all("a", href=True):
        if any(x in a["href"].lower() for x in ["_photo_gallery", "_gallery", "/gallery", "render", "artwork"]):
            gallery_url = urljoin(base_url, a["href"])
            if gallery_url not in gallery_urls:
                gallery_urls.append(gallery_url)
                print(f"[INFO] Found gallery: {gallery_url}")
                try:
                    gallery_response = requests.get(gallery_url, headers=headers)
                    gallery_soup = BeautifulSoup(gallery_response.text, "html.parser")
                    gallery_imgs = gallery_soup.find_all("img", {"src": True})
                    print(f"[INFO] Found {len(gallery_imgs)} images in gallery")
                    imgs.extend(gallery_imgs)
                except Exception as e:
                    print(f"[WARN] Gallery error: {e}")

    # 2. Check main page images
    main_imgs = soup.find_all("img", {"src": True})
    print(f"[INFO] Found {len(main_imgs)} images on main page")
    imgs.extend(main_imgs)

    # Filter and process images
    valid_imgs = []
    seen_urls = set()

    for img in imgs:
        src = img.get("src", "").split("?")[0]  # Remove URL parameters
        if not src or src in seen_urls:
            continue

        # Convert to full resolution URL
        if "/thumb/" in src:
            parts = src.split("/thumb/")
            if len(parts) == 2:
                # Remove the thumbnail size specification
                src = parts[0] + "/" + "/".join(parts[1].split("/")[:-1])

        # Skip small images and unwanted types
        width = int(img.get("width", "0"))
        if width and width < 200:
            continue
            
        if any(x in src.lower() for x in ["icon", "sprite", "logo", "button"]):
            continue

        seen_urls.add(src)
        valid_imgs.append(img)

    print(f"[INFO] Found {len(valid_imgs)} valid images after filtering")

    if not valid_imgs:
        print(f"[WARN] No suitable images found for {monster_name}")
        return 0

    # Save the images
    return save_images(valid_imgs, output_folder, monster_name, base_url, headers)


# -------------------------------------------------------
# Main Driver
# -------------------------------------------------------
def scrape_gallery_images(categories, base_folder="MonsterImages"):
    """Main function to scrape images for all monsters by category."""
    manifest_path = os.path.join(base_folder, "manifest.json")
    manifest = load_manifest(manifest_path)
    
    for category, url in categories.items():
        print(f"\nProcessing category: {category}")
        manifest.setdefault(category, {})
        
        # Get monster list
        monsters = get_monsters_by_category(url)
        
        # Process each monster
        for monster in monsters:
            if monster in manifest[category]:
                print(f"Skipping {monster} (already processed)")
                continue
                
            output_folder = os.path.join(base_folder, category, sanitize_name(monster))
            try:
                count = download_gallery_images(monster, output_folder)
                manifest[category][monster] = count
                save_manifest(manifest_path, manifest)
            except Exception as e:
                print(f"Error processing {monster}: {e}")
                manifest[category][monster] = 0
                save_manifest(manifest_path, manifest)
    
    return manifest


# -------------------------------------------------------
# Example Usage
# -------------------------------------------------------
if __name__ == "__main__":
    # Define monster categories and their URLs
    categories = {
        "Amphibians": "https://monsterhunter.fandom.com/wiki/Amphibian",
        "Carapaceons": "https://monsterhunter.fandom.com/wiki/Carapaceon",
        "Piscine_Wyverns": "https://monsterhunter.fandom.com/wiki/Piscine_Wyvern",
        "Fanged_Beasts": "https://monsterhunter.fandom.com/wiki/Fanged_Beast",
        "Leviathans": "https://monsterhunter.fandom.com/wiki/Leviathan",
        "Temnocerans": "https://monsterhunter.fandom.com/wiki/Temnoceran",
    }

    # Set up base folder for all images
    base_folder = "MonsterImages"
    
    try:
        # Start scraping
        print("Starting monster image scraping...")
        manifest = scrape_gallery_images(categories, base_folder)
        
        # Print summary
        print("\n=== Scraping Summary ===")
        total_monsters = 0
        total_images = 0
        
        for category, monsters in manifest.items():
            category_count = sum(count for count in monsters.values() if count > 0)
            monster_count = len([m for m in monsters.values() if m > 0])
            print(f"{category}: {monster_count} monsters, {category_count} images")
            total_monsters += monster_count
            total_images += category_count
            
        print(f"\nTotal: {total_monsters} monsters, {total_images} images")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise