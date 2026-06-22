import cv2
import numpy as np
import os
from skimage import io

def crop_fixed_square_around_lesion(img, crop_size=2048, pad=30):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Center of lesion bounding box
    cx = x + w // 2
    cy = y + h // 2

    half_crop = crop_size // 2

    # Calculate crop coordinates
    x_start = cx - half_crop
    y_start = cy - half_crop
    x_end = cx + half_crop
    y_end = cy + half_crop

    # Pad image if crop is out of bounds
    top_pad = max(0, -y_start)
    left_pad = max(0, -x_start)
    bottom_pad = max(0, y_end - img.shape[0])
    right_pad = max(0, x_end - img.shape[1])

    # Apply padding if needed
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Adjust coordinates after padding
    y_start += top_pad
    y_end += top_pad
    x_start += left_pad
    x_end += left_pad

    crop = img[y_start:y_end, x_start:x_end]

    # Create mask for crop area (all white)
    mask = 255 * np.ones((crop_size, crop_size), dtype=np.uint8)

    bbox = (x_start, y_start, crop_size, crop_size)
    return crop, mask, bbox


def process_directory_fixed_crop(input_dir, cropped_dir, overlay_dir, crop_size=2048):
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            img_path = os.path.join(input_dir, fname)
            img = io.imread(img_path)

            cropped, mask, bbox = crop_fixed_square_around_lesion(img, crop_size=crop_size)

            if cropped is None:
                print(f"No lesion found in {fname}, skipping.")
                continue

            # Save cropped image
            cropped_path = os.path.join(cropped_dir, fname)
            cv2.imwrite(cropped_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

            # Draw bounding box overlay on original image
            x, y, w, h = bbox
            overlay_img = img.copy()

            # If padding was applied, overlay coordinates may not align with original image.
            # To show overlay on original image without padding, draw lesion bbox (approximate).
            # So here, draw the lesion bounding box:
            lesion_x, lesion_y, lesion_w, lesion_h = cv2.boundingRect(max(cv2.findContours(
                cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                              0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea))
            cv2.rectangle(overlay_img, (lesion_x, lesion_y), (lesion_x + lesion_w, lesion_y + lesion_h), (255, 0, 0), 3)

            overlay_path = os.path.join(overlay_dir, fname)
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

            print(f"Processed {fname}")


if __name__ == "__main__":
    input_dir = "path/to/your/input_images"
    cropped_dir = "path/to/save/cropped_images"
    overlay_dir = "path/to/save/overlay_images"

    process_directory_fixed_crop(input_dir, cropped_dir, overlay_dir, crop_size=2048)


if __name__ == "__main__":
    input_dir = "/mnt/d/Naved/Data/ISIC100/data/hr/all"          # your input folder
    cropped_dir = "/mnt/d/Naved/Data/ISIC100/data/hr_cropped/all"    # your output folder
    overlay_dir = "/mnt/d/Naved/Data/ISIC100/data/hr_cropped/overlay_images"

    process_directory(input_dir, cropped_dir, overlay_dir, pad=30)
