"""
convert_coco_to_yolo.py
=======================
Step 2 of dataset preparation for:
"AI-Based Underwater Visual Perception for Real-Time ROV Object Detection"

WHAT THIS SCRIPT DOES
---------------------
It reads the ORIGINAL COCO file (`dataset.json`) and the split plan made by
`create_split.py`, then builds a brand new YOLO dataset:

    yolo_dataset/
        images/train   labels/train
        images/val     labels/val
        images/test    labels/test
        dataset.yaml

    cross_domain_test/
        images/   labels/
        dataset.yaml

Two conversions happen here.

  1. BOUNDING BOX
     COCO stores  [x, y, width, height]  in PIXELS.
     YOLO wants  [class_id, centre_x, centre_y, width, height]  NORMALISED 0..1.

        centre_x = (x + width  / 2) / image_width
        centre_y = (y + height / 2) / image_height
        new_width  = width  / image_width
        new_height = height / image_height

     The image width and height are read from dataset.json for each image, so
     nothing is hard-coded to 1920x1080.

  2. CLASS ID
     COCO uses 1..40, YOLO must use 0..39.

        yolo_class_id = coco_category_id - 1

SAFETY
------
The original dataset is opened READ ONLY. Every new file is written into the
new `yolo_dataset` and `cross_domain_test` folders. Nothing in the original
`<Site>/<Camera>` folders is changed, renamed, moved or deleted.

HOW TO RUN
----------
    python scripts/create_split.py            (run this first)
    python scripts/convert_coco_to_yolo.py
"""

import json
import os
import shutil
import collections

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

ORIGINAL_ROOT = r"C:\Nehal uiu\AI Vission"
COCO_JSON = os.path.join(ORIGINAL_ROOT, "dataset.json")

YOLO_FOLDER = os.path.join(ORIGINAL_ROOT, "yolo_dataset")
CROSS_DOMAIN_FOLDER = os.path.join(ORIGINAL_ROOT, "cross_domain_test")

# The plan file written by create_split.py
PLAN_JSON = os.path.join(YOLO_FOLDER, "split_plan.json")

# Files we are allowed to write.
FILENAME_MAPPING_CSV = os.path.join(YOLO_FOLDER, "filename_mapping.csv")

# The main pool of splits that make up the 70 / 15 / 15 split.
MAIN_SPLITS = ["train", "val", "test"]


# ----------------------------------------------------------------------------
# HELPER 1: convert one COCO box into a YOLO box
# ----------------------------------------------------------------------------
def convert_box(coco_box, image_width, image_height):
    """
    Convert one COCO bounding box into YOLO format.

    coco_box    = [x, y, width, height]  in pixels
    image_width / image_height are read from dataset.json

    Returns [centre_x, centre_y, width, height] all normalised to 0..1,
    or None if the box is not usable.
    """
    x, y, width, height = coco_box

    # A box with no area cannot be converted.
    if width <= 0 or height <= 0:
        return None

    centre_x = (x + width / 2.0) / image_width
    centre_y = (y + height / 2.0) / image_height
    new_width = width / image_width
    new_height = height / image_height

    return [centre_x, centre_y, new_width, new_height]


# ----------------------------------------------------------------------------
# HELPER 2: create the folder structure
# ----------------------------------------------------------------------------
def make_folders():
    """Create every output folder we are going to need."""
    print("Creating output folders ...")

    for split in MAIN_SPLITS:
        images_folder = os.path.join(YOLO_FOLDER, "images", split)
        labels_folder = os.path.join(YOLO_FOLDER, "labels", split)
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)
        print("  ", images_folder)

    images_folder = os.path.join(CROSS_DOMAIN_FOLDER, "images")
    labels_folder = os.path.join(CROSS_DOMAIN_FOLDER, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    print("  ", images_folder)


# ----------------------------------------------------------------------------
# HELPER 3: write one YOLO label file
# ----------------------------------------------------------------------------
def write_label_file(label_path, yolo_lines):
    """
    Write one image's label file.

    An image that has no objects still gets an EMPTY .txt file.
    That is the correct YOLO behaviour: the file existing but being empty
    tells YOLO "this image genuinely contains no objects".
    """
    with open(label_path, "w", encoding="utf-8") as handle:
        for line in yolo_lines:
            handle.write(line + "\n")


# ----------------------------------------------------------------------------
# HELPER 4: write dataset.yaml
# ----------------------------------------------------------------------------
def write_dataset_yaml(file_path, root_folder, names, use_test=True):
    """
    Write an Ultralytics YOLO dataset.yaml.

    Ultralytics expects:
        path  - the dataset root folder
        train - folder of training images, relative to path
        val   - folder of validation images
        test  - folder of test images
        names - class id -> class name
    """
    # Ultralytics prefers forward slashes in yaml paths, even on Windows.
    root_with_slashes = root_folder.replace("\\", "/")

    lines = []
    lines.append("# Ultralytics YOLO dataset configuration")
    lines.append("# Generated by scripts/convert_coco_to_yolo.py")
    lines.append("# Source: Seaclear Marine Debris Dataset (COCO -> YOLO conversion)")
    lines.append("")
    lines.append("path: {}".format(root_with_slashes))
    lines.append("train: images/train")
    lines.append("val: images/val")
    if use_test:
        lines.append("test: images/test")
    lines.append("")
    lines.append("# {} classes. COCO ids 1-40 were remapped to YOLO ids 0-39.".format(len(names)))
    lines.append("names:")
    for class_id in range(len(names)):
        lines.append("  {}: {}".format(class_id, names[class_id]))

    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("Wrote", file_path)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 2: CONVERT COCO TO YOLO FORMAT")
    print("=" * 70)

    # ---- load the original COCO file (read only) -------------------------
    print("\nReading original COCO file:", COCO_JSON)
    with open(COCO_JSON, "r", encoding="utf-8") as handle:
        coco = json.load(handle)
    print("  images     :", len(coco["images"]))
    print("  annotations:", len(coco["annotations"]))
    print("  categories :", len(coco["categories"]))

    # ---- load the split plan --------------------------------------------
    if not os.path.exists(PLAN_JSON):
        print("\nERROR: split plan not found.")
        print("Please run this first:  python scripts/create_split.py")
        return

    print("\nReading split plan:", PLAN_JSON)
    with open(PLAN_JSON, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    print("  planned images:", len(plan["assignments"]))
    print("  class names   :", len(plan["class_names"]))

    class_names = plan["class_names"]

    # ---- group the COCO annotations by image ----------------------------
    # This lets us find an image's boxes in one quick lookup.
    print("\nGrouping annotations by image ...")
    annotations_by_image = collections.defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    print("  done")

    # ---- create the folders ---------------------------------------------
    print()
    make_folders()

    # ---- copy every image and write its label file -----------------------
    print("\nCopying images and writing label files ...")

    total_images = 0
    total_label_lines = 0
    images_per_split = collections.Counter()
    lines_per_split = collections.Counter()
    skipped_boxes = 0
    mapping_rows = []

    for position, item in enumerate(plan["assignments"]):
        split = item["split"]
        image_id = item["image_id"]
        old_path = item["old_path"]
        new_name = item["new_name"]

        # Decide which folder this image belongs to.
        if split == "cross_domain_test":
            images_folder = os.path.join(CROSS_DOMAIN_FOLDER, "images")
            labels_folder = os.path.join(CROSS_DOMAIN_FOLDER, "labels")
        else:
            images_folder = os.path.join(YOLO_FOLDER, "images", split)
            labels_folder = os.path.join(YOLO_FOLDER, "labels", split)

        # Copy the image with its new, safe name.
        destination_image = os.path.join(images_folder, new_name)
        shutil.copy2(old_path, destination_image)

        # Build the YOLO lines for this image.
        # Image size is read from dataset.json, not hard-coded.
        image_width = item["width"]
        image_height = item["height"]
        yolo_lines = []

        for annotation in annotations_by_image[image_id]:
            yolo_box = convert_box(annotation["bbox"], image_width, image_height)
            if yolo_box is None:
                skipped_boxes += 1
                continue

            # COCO id 1..40  ->  YOLO id 0..39
            yolo_class_id = annotation["category_id"] - 1

            centre_x, centre_y, box_width, box_height = yolo_box
            line = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                yolo_class_id, centre_x, centre_y, box_width, box_height)
            yolo_lines.append(line)

        # Write the label file next to it.
        label_name = os.path.splitext(new_name)[0] + ".txt"
        destination_label = os.path.join(labels_folder, label_name)
        write_label_file(destination_label, yolo_lines)

        # Keep a record so the rename can always be undone.
        mapping_rows.append("{},{},{},{},{}".format(
            item["subset"], item["old_name"], new_name, split, len(yolo_lines)))

        total_images += 1
        total_label_lines += len(yolo_lines)
        images_per_split[split] += 1
        lines_per_split[split] += len(yolo_lines)

        # Progress message every 1000 images.
        if total_images % 1000 == 0:
            print("  ...", total_images, "images done")

    print("\n  finished", total_images, "images and", total_label_lines, "label lines")
    if skipped_boxes:
        print("  WARNING: skipped", skipped_boxes, "boxes with zero width or height")

    # ---- save the filename mapping ---------------------------------------
    print("\nWriting filename mapping:", FILENAME_MAPPING_CSV)
    with open(FILENAME_MAPPING_CSV, "w", encoding="utf-8") as handle:
        handle.write("subset,original_name,new_name,split,label_lines\n")
        for row in mapping_rows:
            handle.write(row + "\n")
    print("  ", len(mapping_rows), "rows")

    # ---- write the two dataset.yaml files -------------------------------
    print()
    write_dataset_yaml(os.path.join(YOLO_FOLDER, "dataset.yaml"),
                       YOLO_FOLDER, class_names, use_test=True)

    # The cross-domain folder has no train split, so it only needs val and
    # names. Ultralytics can still evaluate on it.
    cross_yaml = os.path.join(CROSS_DOMAIN_FOLDER, "dataset.yaml")
    root_with_slashes = CROSS_DOMAIN_FOLDER.replace("\\", "/")
    lines = []
    lines.append("# Cross-domain test set (WHOLE site-camera subset held out)")
    lines.append("# Selected subset: " + ", ".join(plan["cross_domain_subsets"]))
    lines.append("# These images are never used for training or model selection.")
    lines.append("# Use for the final generalisation number only.")
    lines.append("")
    lines.append("path: {}".format(root_with_slashes))
    lines.append("val: images")
    lines.append("")
    lines.append("names:")
    for class_id in range(len(class_names)):
        lines.append("  {}: {}".format(class_id, class_names[class_id]))
    with open(cross_yaml, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("Wrote", cross_yaml)

    # ---- print a summary -------------------------------------------------
    print()
    print("=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print("Images copied      :", total_images)
    print("Label lines written:", total_label_lines)
    print("Classes            :", len(class_names), "(YOLO ids 0 to", len(class_names) - 1, ")")
    print()
    print("Images per split:")
    for split in MAIN_SPLITS + ["cross_domain_test"]:
        print("  {:18s} {:6d} images, {:6d} label lines".format(
            split, images_per_split[split], lines_per_split[split]))
    print()
    print("Next step:  python scripts/verify_dataset.py")


if __name__ == "__main__":
    main()
