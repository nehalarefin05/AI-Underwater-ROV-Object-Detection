"""
train.py

Simple YOLO training script.

All the settings you might want to change are in the block of constants
at the top of the file. Nothing else needs editing.

Run:
    python scripts/train.py
"""

# ---------------------------------------------------------------------------
# SETTINGS  (change these only)
# ---------------------------------------------------------------------------

MODEL = "yolo26n.pt"             # pretrained model file
DATA = "yolo_dataset/dataset.yaml"  # dataset description
EPOCHS = 1                      # how many times to go through the images
IMGSZ = 320                     # image size (must be a multiple of 32)
BATCH = 1                       # images per training step
WORKERS = 0                     # extra data loading processes
DEVICE = "cpu"                  # "cpu" or "0" for the first GPU

# Where the results are saved. Each run gets its own numbered folder.
PROJECT = "runs"
NAME = "yolo26n_baseline"

# Print and save this fraction of the training data.
# 1.0 = use all of it.  0.05 = use only 5%.
FRACTION = 1.0

# Keep images in memory between epochs. False = slower but uses less RAM.
CACHE = False

# Stop early if the model stops improving for this many epochs.
PATIENCE = 20

# ---------------------------------------------------------------------------

import os

from ultralytics import YOLO


def check_dataset_files():
    """Make sure the dataset YAML exists and the split folders are present."""
    if not os.path.exists(DATA):
        print("ERROR: dataset YAML not found:", DATA)
        print("Run scripts/convert_coco_to_yolo.py first.")
        return False

    base = os.path.dirname(DATA)
    for split in ["train", "val"]:
        folder = os.path.join(base, "images", split)
        if not os.path.isdir(folder):
            print("ERROR: missing folder:", folder)
            return False

    return True


def check_class_count():
    """Count the class names listed in the dataset YAML."""
    class_count = 0
    in_names = False

    with open(DATA, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if line == "":
                    break
                if ":" in line:
                    class_count += 1

    return class_count


def show_configuration():
    """Print the settings so they are visible in the log."""
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print("  model       :", MODEL)
    print("  dataset     :", DATA)
    print("  epochs      :", EPOCHS)
    print("  image size  :", IMGSZ)
    print("  batch       :", BATCH)
    print("  workers     :", WORKERS)
    print("  device      :", DEVICE)
    print("  fraction    :", FRACTION)
    print("  cache       :", CACHE)
    print("  patience    :", PATIENCE)
    print("  output      :", os.path.join(PROJECT, NAME))
    print("=" * 60)
    print("")


def main():
    if not check_dataset_files():
        return

    class_count = check_class_count()
    show_configuration()
    print("Number of classes in dataset YAML:", class_count)
    print("")

    # Load the pretrained model.
    model = YOLO(MODEL)

    # Train.
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=False,   # False = never overwrite an earlier run
        fraction=FRACTION,
        cache=CACHE,
        patience=PATIENCE,
    )

    print("")
    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    save_dir = model.trainer.save_dir
    print("Results saved to :", save_dir)
    print("Best weights     :", os.path.join(save_dir, "weights", "best.pt"))
    print("Last weights     :", os.path.join(save_dir, "weights", "last.pt"))
    print("=" * 60)


if __name__ == "__main__":
    main()
