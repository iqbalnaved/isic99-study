import os
import requests
from tqdm import tqdm
import json

def get_image_ids_from_directory(image_dir):
    """Extract ISIC image IDs from image filenames."""
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return [os.path.splitext(f)[0].strip() for f in image_files if f.startswith("ISIC_")]

def fetch_full_clinical_metadata(image_ids, output_dir="clinical_jsons"):
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://api.isic-archive.com/api/v2/images/{}"

    for image_id in tqdm(image_ids, desc="Downloading clinical JSONs"):
        url = base_url.format(image_id)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                json_path = os.path.join(output_dir, f"{image_id}.json")
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                print(f"[{image_id}] Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"[{image_id}] Error: {e}")

# === Main Execution ===
if __name__ == "__main__":
    image_dir = "/mnt/d/Naved/Data/ISIC100/data/all"  # <-- change this
    save_path = "/mnt/d/Naved/Data/ISIC100/clinical_data"
    image_ids = get_image_ids_from_directory(image_dir)
    fetch_full_clinical_metadata(image_ids, output_dir=save_path)
