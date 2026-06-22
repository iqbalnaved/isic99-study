import os
import re
import json
import pandas as pd
from tqdm import tqdm

def extract_metadata_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    clinical = meta.get("clinical", {})
    acquisition = meta.get("acquisition", {})
    full_image = data.get("files", {}).get("full", {})

    # Extract all diagnosis fields dynamically
    diagnosis_fields = {
        k: v for k, v in clinical.items() if re.match(r"diagnosis_\d+", k)
    }

    # Basic metadata
    metadata = {
        "isic_id": data.get("isic_id"),
        "sex": clinical.get("sex"),
        "age_approx": clinical.get("age_approx"),
        "anatom_site_general": clinical.get("anatom_site_general"),
        "diagnosis_confirm_type": clinical.get("diagnosis_confirm_type"),
        "melanocytic": clinical.get("melanocytic"),
        "concomitant_biopsy": clinical.get("concomitant_biopsy"),
        "lesion_id": clinical.get("lesion_id"),
        "image_type": acquisition.get("image_type"),
        "dermoscopic_type": acquisition.get("dermoscopic_type"),
        "pixels_x": acquisition.get("pixels_x"),
        "pixels_y": acquisition.get("pixels_y"),
        "image_url": full_image.get("url"),
        "image_size": full_image.get("size")
    }

    metadata.update(diagnosis_fields)
    return metadata

def jsons_to_csv(json_dir, output_csv="isic_metadata_with_diagnosis.csv"):
    all_records = []
    all_diagnosis_keys = set()
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            record = extract_metadata_from_json(os.path.join(json_dir, json_file))
            all_records.append(record)
            # Track all unique diagnosis_N keys
            all_diagnosis_keys.update([k for k in record.keys() if re.match(r"diagnosis_\d+", k)])
        except Exception as e:
            print(f"Failed to process {json_file}: {e}")

    # Ensure all rows have all diagnosis columns
    all_columns = set().union(*[r.keys() for r in all_records])
    all_columns.update(all_diagnosis_keys)  # ensure missing diagnosis_N are added

    df = pd.DataFrame(all_records, columns=sorted(all_columns))
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved metadata CSV to: {output_csv}")


# === Main ===
if __name__ == "__main__":
    json_dir = "/mnt/d/Naved/Data/ISIC100/clinical_data"  # <-- update this
    jsons_to_csv(json_dir, output_csv="/mnt/d/Naved/Data/ISIC100/isic100_clinical_metadata.csv")
