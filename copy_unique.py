import os
import shutil

def copy_unique_images(src_dir, exclude_dir, dest_dir, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
    """
    Copy images from src_dir to dest_dir if they are not present in exclude_dir.
    
    Args:
        src_dir (str): Path to the source directory.
        exclude_dir (str): Path to the directory containing images to exclude.
        dest_dir (str): Path to the destination directory.
        exts (tuple): Allowed image extensions.
    """
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Collect excluded filenames
    exclude_files = {f for f in os.listdir(exclude_dir) if f.lower().endswith(exts)}

    # Iterate over source directory
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(exts) and filename not in exclude_files:
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)

            # Copy the file
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {filename}")

    print("✅ Copying complete.")

# Example usage
if __name__ == "__main__":
    source = "/mnt/d/Naved/Data/ISIC100/data/originals/all"
    exclude = "/mnt/d/Naved/Data/ISIC100/data/originals/bn"
    destination = "/mnt/d/Naved/Data/ISIC100/data/originals/mm"

    copy_unique_images(source, exclude, destination)
