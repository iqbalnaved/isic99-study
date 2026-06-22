# crop ROI with SAM

import torch
import numpy as np
import cv2
import os
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load SAM model
def load_sam_model(checkpoint_path, model_type="vit_b"):
    print(f"[INFO] Loading SAM model ({model_type}) from: {checkpoint_path}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # mask_generator = SamAutomaticMaskGenerator(
        # sam,
        # points_per_side=64,  # increase for more precise masks (default: 32)
        # pred_iou_thresh=0.9, # increase to discard uncertain masks
        # stability_score_thresh=0.95, # increase to keep only stable masks
        # crop_n_layers=0,  # set to 0 to avoid subpatch splitting
        # min_mask_region_area=500   # discard small blobs. adjust based on image resolution
    # )

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,  # increase for more precise masks (default: 32)
        pred_iou_thresh=0.9, # increase to discard uncertain masks
        stability_score_thresh=0.95, # increase to keep only stable masks
        crop_n_layers=0,  # set to 0 to avoid subpatch splitting
        min_mask_region_area=500   # discard small blobs. adjust based on image resolution
    )

    return mask_generator

# Overlay and save largest mask
def save_mask_overlay(image, mask, save_path, color=(0, 0, 255), alpha=0.4):
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[:, :] = color
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)[mask_bool]
    cv2.imwrite(str(save_path), overlay)

def get_central_mask(masks, image_shape):
    h, w = image_shape[:2]
    center = np.array([w // 2, h // 2])

    best_mask = None
    min_distance = float('inf')

    for m in masks:
        x, y, bw, bh = m['bbox']
        centroid = np.array([x + bw / 2, y + bh / 2])
        dist = np.linalg.norm(centroid - center)

        if dist < min_distance:
            min_distance = dist
            best_mask = m

    return best_mask

# Crop largest mask and resize
def crop_largest_mask(image, masks, desired_size=(2048, 2048), padding=10):
    if len(masks) == 0:
        print("[DEBUG] No masks found.")
        return None, None

    #largest_mask = max(masks, key=lambda x: x['area'])
    # mask = largest_mask['segmentation'].astype(np.uint8) * 255
    
    central_mask = get_central_mask(masks, image.shape)
    mask = central_mask['segmentation'].astype(np.uint8) * 255
    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[DEBUG] No contours found in mask.")
        return None, None

    x, y, w, h = cv2.boundingRect(contours[0])
    print(f"[DEBUG] Bounding box: x={x}, y={y}, w={w}, h={h}")

    x = max(x - padding, 0)
    y = max(y - padding, 0)
    x2 = min(x + w + 2 * padding, image.shape[1])
    y2 = min(y + h + 2 * padding, image.shape[0])
    cropped = image[y:y2, x:x2]

    print(f"[DEBUG] Cropped size: {cropped.shape[1]}x{cropped.shape[0]}")
    resized = cv2.resize(cropped, desired_size, interpolation=cv2.INTER_CUBIC)
    print(f"[DEBUG] Resized to: {desired_size[0]}x{desired_size[1]}")
    return resized, mask

def get_central_mask(masks, image_shape):
    h, w = image_shape[:2]
    center = np.array([w // 2, h // 2])

    best_mask = None
    min_distance = float('inf')

    for m in masks:
        x, y, bw, bh = m['bbox']
        centroid = np.array([x + bw / 2, y + bh / 2])
        dist = np.linalg.norm(centroid - center)

        if dist < min_distance:
            min_distance = dist
            best_mask = m

    return best_mask


# Process images with overlays and debug info
def process_images_with_sam(input_dir, output_dir, sam_checkpoint, desired_size=(2048, 2048)):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    overlay_dir = output_dir / "mask_overlay"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return

    image_paths = list(input_dir.glob("*.jpg"))
    print(f"[INFO] Found {len(image_paths)} image(s) in: {input_dir}")

    if len(image_paths) == 0:
        print("[WARNING] No .jpg images found. Check the directory or file extensions.")
        return

    mask_generator = load_sam_model(sam_checkpoint)

    for i, img_path in enumerate(image_paths):
        print(f"\n[INFO] Processing image {i + 1}/{len(image_paths)}: {img_path.name}")
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"[ERROR] Failed to read image: {img_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_rgb)

        print(f"[DEBUG] SAM generated {len(masks)} mask(s)")

        cropped, mask = crop_largest_mask(image, masks, desired_size=desired_size)

        if cropped is not None:
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), cropped)
            print(f"[INFO] Saved cropped image to: {out_path}")

            # Save overlay
            overlay_path = overlay_dir / img_path.name
            save_mask_overlay(image, mask, overlay_path)
            print(f"[INFO] Saved mask overlay to: {overlay_path}")
        else:
            print(f"[WARNING] Skipping {img_path.name} (no valid ROI found)")

# if __name__ == "__main__":
    # input_dir = r"D:\Naved\Data\ISIC100\data\hr_cropped\all"
    # output_dir = r"D:\Naved\Data\ISIC100\data\roi_sam_2048"
    # sam_checkpoint = "sam_vit_b_01ec64.pth"
    # desired_crop_size = (2048, 2048)

    # process_images_with_sam(
        # input_dir=input_dir,
        # output_dir=output_dir,
        # sam_checkpoint=sam_checkpoint,
        # desired_size=desired_crop_size
    # )




if __name__ == "__main__":
    input_dir = "/mnt/d/Naved/Data/ISIC100/data/hr/all"          # your input folder
    output_dir = "/mnt/d/Naved/Data/ISIC100/data/hr_cropped/all"    # your output folder
    sam_checkpoint = "/mnt/d/Naved/Codes/segment-anything/models/sam_vit_b_01ec64.pth"
    desired_crop_size = (2048, 2048)      # <-- specify your desired size here

    process_images_with_sam(
        input_dir=input_dir,
        output_dir=output_dir,
        sam_checkpoint=sam_checkpoint,
        desired_size=desired_crop_size
    )
