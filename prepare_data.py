import os
from PIL import Image

SOURCE_DIR = "/app/hanabi_dataset"
TARGET_DIR = "/app/hanabi_dataset_small"
TARGET_SIZE = (800, 800)

def resize_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory {SOURCE_DIR} not found.")
        return

    # 1. Scan for all valid image files
    all_files = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))

    total_found = len(all_files)
    
    # 2. Filter for only images that NEED resizing
    to_process = []
    for src_path in all_files:
        rel_path = os.path.relpath(src_path, SOURCE_DIR)
        target_path = os.path.join(TARGET_DIR, rel_path)
        if not os.path.exists(target_path):
            to_process.append((src_path, target_path))

    total_to_resize = len(to_process)
    already_done = total_found - total_to_resize
    
    print("-" * 30)
    print(f"HANABI DATASET SYNC")
    print(f"Total source images: {total_found}")
    print(f"Already synced:      {already_done}")
    print(f"Pending resize:      {total_to_resize}")
    print("-" * 30)

    if total_to_resize == 0:
        print("Dataset is up to date. No work needed.")
        return

    # 3. Process with simple text progress
    for index, (src_path, target_path) in enumerate(to_process):
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Simple progress update every 10 images or at the end
            if index % 10 == 0 or index + 1 == total_to_resize:
                percent = int(((index + 1) / total_to_resize) * 100)
                print(f"Processing: {index + 1}/{total_to_resize} ({percent}%)")

            with Image.open(src_path) as img:
                img.thumbnail(TARGET_SIZE)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(target_path, "JPEG", quality=85)
                
        except Exception as e:
            print(f" ! Error processing {src_path}: {e}")

    print("-" * 30)
    print(f"SUCCESS: All {total_found} images are ready in {TARGET_DIR}")

if __name__ == "__main__":
    resize_dataset()