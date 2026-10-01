"""
evaluate.py

Simple evaluation script. Loads a trained model and measures it on a
validation dataset.

Run:
    python scripts/evaluate.py

To test a different model or dataset, change the two settings below.
"""

# ---------------------------------------------------------------------------
# SETTINGS  (change these only)
# ---------------------------------------------------------------------------

MODEL = "runs/yolo26n_baseline/weights/best.pt"   # trained model file
DATA = "yolo_dataset/dataset.yaml"                # dataset description
IMGSZ = 320                                       # must match training
BATCH = 1
DEVICE = "cpu"

# ---------------------------------------------------------------------------

import os

from ultralytics import YOLO


def check_files():
    """Stop early with a clear message if something is missing."""
    if not os.path.exists(MODEL):
        print("ERROR: model file not found:", MODEL)
        print("Train the model first with:  python scripts/train.py")
        return False

    if not os.path.exists(DATA):
        print("ERROR: dataset YAML not found:", DATA)
        return False

    return True


def main():
    if not check_files():
        return

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print("  model  :", MODEL)
    print("  dataset:", DATA)
    print("  split  : val (built into the dataset YAML)")
    print("=" * 60)
    print("")

    model = YOLO(MODEL)

    results = model.val(
        data=DATA,
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
    )

    # The results object holds the overall scores.
    box = results.box

    print("")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("  mAP50    :", round(float(box.map50), 4))
    print("  mAP50-95 :", round(float(box.map), 4))
    print("  precision:", round(float(box.mp), 4))
    print("  recall   :", round(float(box.mr), 4))
    print("=" * 60)


if __name__ == "__main__":
    main()
