#!/usr/bin/env python3
"""Utility script to download additional authentic TACO roadside litter images from Flickr."""
import argparse
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "taco_samples"
URLS_FILE = ROOT / "data" / "taco" / "data" / "all_image_urls.csv"

# Pre-curated high quality TACO roadside & storm drain waste photos from Flickr
ADDITIONAL_TACO_PRESETS = [
    ("sample_plastic_cups_gutter.jpg", "https://farm66.staticflickr.com/65535/47855505601_a81c3ba8de_z.jpg", "Disposable plastic cups along street gutter"),
    ("sample_snack_wrappers_curb.jpg", "https://farm66.staticflickr.com/65535/47066066634_60a4f44241_z.jpg", "Multilayer snack packaging near drain inlet"),
    ("sample_cardboard_box_drain.jpg", "https://farm66.staticflickr.com/65535/47803337262_4965d5608b_z.jpg", "Soggy cardboard carton obstructing culvert intake"),
    ("sample_polypropylene_sacks.jpg", "https://farm66.staticflickr.com/65535/47066067064_b7ca7a114d_z.jpg", "Heavy polypropylene sack blocking stormwater flow"),
    ("sample_pet_bottle_pile.jpg", "https://farm66.staticflickr.com/65535/46939252345_2a66fc213a_z.jpg", "Dense pile of clear PET bottles in runoff pathway"),
]


def download_additional_samples():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(ADDITIONAL_TACO_PRESETS)} additional authentic TACO roadside photos...")
    
    count = 0
    for filename, url, desc in ADDITIONAL_TACO_PRESETS:
        dest = SAMPLES_DIR / filename
        if not dest.exists():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=12) as r:
                    dest.write_bytes(r.read())
                print(f"  ✓ Downloaded {filename} ({dest.stat().st_size // 1024} KB) - {desc}")
                count += 1
            except Exception as e:
                print(f"  ✗ Failed {filename}: {e}")
        else:
            print(f"  • Already exists: {filename}")

    print(f"\nDone! Total images in data/taco_samples/: {len(list(SAMPLES_DIR.glob('*.*')))}")


if __name__ == "__main__":
    download_additional_samples()
