#!/usr/bin/env python3
"""
Migrate old color-images JSON files to new folder structure:
  {timestamp}/
    {color}.jpeg
    request.json
"""
import base64
import json
import os
from datetime import datetime, timezone

SAVE_DIR = "/srv/data/bnutfilloyev/file_collector/color-images"


def decode_base64(image_str: str) -> bytes:
    if "base64," in image_str:
        image_str = image_str.split("base64,")[1]
    padding = 4 - len(image_str) % 4
    if padding != 4:
        image_str += "=" * padding
    return base64.urlsafe_b64decode(image_str)


def migrate_file(json_path: str):
    filename = os.path.basename(json_path)
    mtime = os.path.getmtime(json_path)
    timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_dir = os.path.join(SAVE_DIR, timestamp)

    print(f"  {filename} → {timestamp}/")

    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(session_dir, exist_ok=True)

    color_counts: dict[str, int] = {}
    saved_files = []
    total = 0

    for color, images in data.items():
        for image_b64 in images:
            count = color_counts.get(color, 0)
            color_counts[color] = count + 1
            fname = f"{color}.jpeg" if count == 0 else f"{color}_{count}.jpeg"
            image_path = os.path.join(session_dir, fname)
            with open(image_path, "wb") as f:
                f.write(decode_base64(image_b64))
            saved_files.append(fname)
            total += 1

    request_meta = {
        "timestamp": timestamp,
        "migrated_from": filename,
        "total_images": total,
        "colors": list(color_counts.keys()),
        "files": saved_files,
    }
    with open(os.path.join(session_dir, "request.json"), "w") as f:
        json.dump(request_meta, f, ensure_ascii=False, indent=2)

    os.remove(json_path)
    print(f"    saved {total} images, deleted old file")


def main():
    json_files = [
        os.path.join(SAVE_DIR, f)
        for f in os.listdir(SAVE_DIR)
        if f.endswith(".json") and f != "request.json"
    ]

    if not json_files:
        print("No old JSON files found.")
        return

    print(f"Found {len(json_files)} file(s) to migrate:\n")
    for path in json_files:
        migrate_file(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
