"""
monster_hunter_scraper_uc_noclick.py
Stealth Monster Hunter Google Image scraper without clicking thumbnails
Resolves move target out of bounds and click failures
"""

import time
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

# ===== SETUP STEALTH DRIVER =====
def setup_driver(headless=True):
    """Stealth Chrome driver using undetected-chromedriver"""
    options = uc.ChromeOptions()
    if headless:
        options.headless = True
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    driver = uc.Chrome(options=options)
    return driver

# ===== SCROLLING =====
def scroll_to_load_images(driver, pause_time=1, scroll_count=10):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(scroll_count):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# ===== IMAGE DOWNLOAD =====
def download_images(search_query, output_folder, num_images=50):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"🔍 Searching for: {search_query}")

    driver = setup_driver(headless=True)
    search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
    driver.get(search_url)
    time.sleep(3)

    # Handle Google consent popup
    try:
        agree_button = driver.find_element(
            By.XPATH, "//button/div[normalize-space()='I agree' or normalize-space()='Accept all']"
        )
        agree_button.click()
        print("✅ Accepted Google consent popup")
        time.sleep(2)
    except:
        pass

    scroll_to_load_images(driver, pause_time=2, scroll_count=10)

    thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
    if not thumbnails:
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf")

    print(f"📦 Found {len(thumbnails)} thumbnails")
    downloaded = 0
    seen_urls = set()

    for idx, thumb in enumerate(thumbnails):
        if downloaded >= num_images:
            break
        try:
            img_url = thumb.get_attribute("src") or thumb.get_attribute("data-src")
            if not img_url or img_url.startswith("data:") or img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            # Download image
            response = requests.get(
                img_url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            img = Image.open(BytesIO(response.content))
            if img.mode != "RGB":
                img = img.convert("RGB")

            safe_name = search_query.replace(" ", "_").replace("/", "_")
            filename = f"{output_folder}/{safe_name}_{downloaded:03d}.jpg"
            img.save(filename, "JPEG")
            downloaded += 1
            print(f"  ✓ Downloaded {downloaded}/{num_images}")

        except Exception as e:
            print(f"  ⚠️ Skipped image {idx}: {e}")
            continue

    driver.quit()
    print(f"✅ Downloaded {downloaded} images for '{search_query}'\n")
    return downloaded

# ===== MONSTER LIST =====
monsters_by_category = {
    "Piscine Wyverns": [
        "Cephadrome Monster Hunter",
        "Plesioth Monster Hunter",
        "Lavasioth Monster Hunter",
        "Beotodus Monster Hunter",
    ],
    "Neopterons": [
        "Seltas Monster Hunter",
        "Seltas Queen Monster Hunter",
    ],
    "Fanged Wyverns": [
        "Zinogre Monster Hunter",
        "Tobi-Kadachi Monster Hunter",
        "Magnamalo Monster Hunter",
    ],
}

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    base_output = "data/raw_google"
    images_per_monster = 40

    print("=" * 60)
    print("🎮 Stealth Monster Hunter Google Images Scraper (No Click)")
    print("=" * 60)
    print(f"\n📁 Saving to: {base_output}/")
    print(f"🖼️ Images per monster: {images_per_monster}\n")

    total_downloaded = 0
    for category, monsters in monsters_by_category.items():
        print(f"\n{'='*60}")
        print(f"📂 Category: {category}")
        print(f"{'='*60}\n")

        category_folder = f"{base_output}/{category}"
        for monster in monsters:
            downloaded = download_images(
                search_query=monster,
                output_folder=category_folder,
                num_images=images_per_monster,
            )
            total_downloaded += downloaded
            time.sleep(2)  # polite pause between monsters

    print("\n" + "=" * 60)
    print(f"🎉 COMPLETE! Total images downloaded: {total_downloaded}")
    print("=" * 60)
