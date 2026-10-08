"""
make_colab_package.py

Copies the finished project into colab_transfer/ and zips it, ready to
upload to Google Colab.

Rules followed:
  - the original dataset is only READ, never changed
  - .cache and __pycache__ files are left out
  - the original dataset.json is NOT included

Run:
    python scripts/make_colab_package.py
"""

import os
import shutil
import zipfile

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

PROJECT = "."
TRANSFER = "colab_transfer"
ZIP_NAME = "colab_transfer.zip"

# Folders to copy
FOLDERS = ["yolo_dataset", "cross_domain_test", "scripts"]

# Single files to copy
FILES = [
    "DATASET_AUDIT.md",
    "SPLIT_REPORT.md",
    "dataset_summary.json",
]

# filename_mapping.csv lives inside yolo_dataset, but the task asks for it
# at the top level too, so it is copied twice on purpose.
EXTRA_FILES = [
    os.path.join("yolo_dataset", "filename_mapping.csv"),
]

# Never include these
SKIP_NAMES = [".cache", "__pycache__"]
SKIP_SUFFIXES = [".cache", ".pyc"]


def should_skip(name):
    """Return True if this file should not be copied."""
    if name in SKIP_NAMES:
        return True
    for suffix in SKIP_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def copy_one_file(source, destination):
    """Copy a single file, creating the target folder if needed."""
    if not os.path.exists(source):
        print("  MISSING, skipped:", source)
        return False

    folder = os.path.dirname(destination)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    shutil.copy2(source, destination)
    return True


def copy_folder(source, destination):
    """Copy a whole folder, skipping cache files. Returns (files, bytes)."""
    file_count = 0
    byte_count = 0

    for root, dirs, files in os.walk(source):
        # Skip unwanted folders in place.
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]

        for name in files:
            if should_skip(name):
                print("    skipped:", os.path.join(root, name))
                continue

            src_file = os.path.join(root, name)
            rel = os.path.relpath(src_file, source)
            dst_file = os.path.join(destination, rel)

            copy_one_file(src_file, dst_file)
            file_count += 1
            byte_count += os.path.getsize(src_file)

    return file_count, byte_count


def add_folder_to_zip(zip_file, folder):
    """Add a folder to the zip, skipping cache files."""
    count = 0
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        for name in files:
            if should_skip(name):
                continue
            full = os.path.join(root, name)
            zip_file.write(full, full)
            count += 1
    return count


def main():
    # Remove an old package only if it is one we made before.
    if os.path.isdir(TRANSFER):
        print("Removing old package folder:", TRANSFER)
        shutil.rmtree(TRANSFER)
    if os.path.exists(ZIP_NAME):
        print("Removing old zip:", ZIP_NAME)
        os.remove(ZIP_NAME)

    os.makedirs(TRANSFER)

    total_files = 0
    total_bytes = 0

    print("")
    print("Copying folders...")
    for name in FOLDERS:
        source = os.path.join(PROJECT, name)
        destination = os.path.join(TRANSFER, name)
        if not os.path.isdir(source):
            print("  MISSING folder, skipped:", source)
            continue
        count, size = copy_folder(source, destination)
        print("  %-20s %6d files  %8.2f MB" % (name, count, size / (1024 ** 2)))
        total_files += count
        total_bytes += size

    print("")
    print("Copying report files...")
    for name in FILES:
        ok = copy_one_file(
            os.path.join(PROJECT, name),
            os.path.join(TRANSFER, name),
        )
        if ok:
            size = os.path.getsize(os.path.join(TRANSFER, name))
            print("  %-20s %8.2f KB" % (name, size / 1024))
            total_files += 1
            total_bytes += size

    print("")
    print("Copying filename_mapping.csv to the top level...")
    for name in EXTRA_FILES:
        target = os.path.join(TRANSFER, os.path.basename(name))
        if copy_one_file(name, target):
            print("  %-20s %8.2f KB" % (os.path.basename(name),
                                        os.path.getsize(target) / 1024))
            total_files += 1
            total_bytes += os.path.getsize(target)

    print("")
    print("Total copied: %d files, %.2f MB" % (total_files, total_bytes / (1024 ** 2)))

    # -----------------------------------------------------------------
    # Build the zip
    # -----------------------------------------------------------------
    print("")
    print("Creating zip...")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name in FOLDERS:
            folder = os.path.join(TRANSFER, name)
            if os.path.isdir(folder):
                count = add_folder_to_zip(zip_file, folder)
                print("  added %-20s %6d files" % (name, count))
        for name in FILES:
            full = os.path.join(TRANSFER, name)
            if os.path.exists(full):
                zip_file.write(full, full)
                print("  added %-20s" % name)
        mapping = os.path.join(TRANSFER, "filename_mapping.csv")
        if os.path.exists(mapping):
            zip_file.write(mapping, mapping)
            print("  added %-20s" % "filename_mapping.csv")

    zip_mb = os.path.getsize(ZIP_NAME) / (1024 ** 2)
    print("")
    print("=" * 60)
    print("PACKAGE READY")
    print("=" * 60)
    print("  folder :", os.path.abspath(TRANSFER))
    print("  zip    :", os.path.abspath(ZIP_NAME))
    print("  size   : %.2f MB" % zip_mb)
    print("=" * 60)


if __name__ == "__main__":
    main()
