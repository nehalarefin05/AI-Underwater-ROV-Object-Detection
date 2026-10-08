"""
ai_vision_app.py

Simple local interface for the project:
"AI-Based Underwater Visual Perception for Real-Time ROV Object Detection"

It offers three input modes:
    1. Laptop Camera   - live camera feed with detection
    2. Video File      - detect on a video, frame by frame
    3. Image File      - detect on a single picture

Run:
    python scripts/ai_vision_app.py
"""

# ---------------------------------------------------------------------------
# SETTINGS  (change these only)
# ---------------------------------------------------------------------------

MODEL_PATH = "best.pt"     # the trained YOLO model

CONFIDENCE = 0.25          # only show boxes above this score (0 to 1)
IMGSZ = 640                # detection image size

DEVICE = "cpu"             # "cpu" or "gpu". "auto" picks for you.
# DEVICE = "auto"        # uncomment this to choose automatically

# ---------------------------------------------------------------------------

import os
import time

import cv2
import torch
from ultralytics import YOLO

# Colours in BGR order, which is what OpenCV uses
BOX_COLOUR = (0, 255, 0)
TEXT_COLOUR = (255, 255, 255)
STATUS_COLOUR = (0, 255, 255)

FONT = 0                   # 0 = FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


# ===========================================================================
# SMALL HELPER FUNCTIONS
# ===========================================================================

def choose_device():
    """
    Return "gpu" if an NVIDIA CUDA GPU is available, otherwise "cpu".
    This way the same program runs on a laptop and on a GPU machine.
    """
    try:
        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


def load_model(device):
    """Load the trained model, or print a helpful message and stop."""
    if not os.path.exists(MODEL_PATH):
        print("")
        print("ERROR: model file not found:", MODEL_PATH)
        print("")
        print("The trained model does not exist yet.")
        print("Either:")
        print("  1. finish training first with:  python scripts/train.py")
        print("  2. or set MODEL_PATH at the top of this script")
        print("     to the real location of best.pt")
        return None

    print("Loading model:", MODEL_PATH)
    print("Device       :", device)
    model = YOLO(MODEL_PATH)
    print("Model loaded, ready to detect.")
    print("")
    return model


def draw_box(frame, box, class_name, score):
    """Draw one bounding box with its class name and confidence."""
    left = int(box[0])
    top = int(box[1])
    right = int(box[2])
    bottom = int(box[3])

    cv2.rectangle(frame, (left, top), (right, bottom), BOX_COLOUR, 2)

    label = class_name + "  " + str(round(float(score), 2))

    cv2.putText(frame, label, (left, top - 8),
                FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICKNESS)


def draw_info(frame, count, fps, inference_ms):
    """
    Draw the information panel in the top-left corner:
    status, FPS, inference time, object count.
    """
    height = 150
    width = 330

    # A dark semi-transparent panel so the text is readable on any video.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, "AI VISION ACTIVE", (12, 28),
                FONT, 0.75, STATUS_COLOUR, 2)
    cv2.putText(frame, "Objects   : " + str(count),
                (12, 58), FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICKNESS)
    cv2.putText(frame, "FPS       : " + str(round(fps, 1)),
                (12, 82), FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICKNESS)
    cv2.putText(frame, "Inference : " + str(round(inference_ms, 1)) + " ms",
                (12, 106), FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICKNESS)
    cv2.putText(frame, "Press Q to stop", (12, 138),
                FONT, 0.55, (200, 200, 200), 1)

    return frame


def read_one_frame(model, device, frame):
    """
    Run detection on a single frame.
    Returns the drawn frame, the number of objects, and the inference ms.
    """
    start = time.time()

    result = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        device=device,
        verbose=False,
    )[0]

    inference_ms = (time.time() - start) * 1000.0

    count = 0
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()
            class_name = result.names[class_id]

            draw_box(frame, coordinates, class_name, score)
            count += 1

    return frame, count, inference_ms


def ask_save_path(default_name, extensions):
    """
    Ask the user where to save the result.
    Returns "" if the user presses Enter, meaning do not save.
    """
    print("")
    print("Save the result? Enter a file name, or press Enter to skip: ", end="")
    answer = input().strip()

    if answer == "":
        return ""

    # Add the correct file ending if the user forgot it.
    if not answer.lower().endswith(tuple(extensions)):
        answer = answer + extensions[0]

    return answer


# ===========================================================================
# MODE 1: LAPTOP CAMERA
# ===========================================================================

def run_camera(model, device):
    """Detect on the live laptop camera until the user presses Q."""
    print("")
    print("=" * 60)
    print("MODE 1 - LAPTOP CAMERA")
    print("=" * 60)
    print("Press Q in the video window to stop.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("")
        print("ERROR: cannot open the laptop camera.")
        print("Close any other app using the camera and try again.")
        return

    frame_number = 0
    start_time = time.time()
    last_count = 0
    last_ms = 0.0

    while True:
        grabbed, frame = camera.read()

        if not grabbed:
            break

        frame, last_count, last_ms = read_one_frame(model, device, frame)

        # Live FPS from the number of frames done so far.
        frame_number += 1
        elapsed = time.time() - start_time
        fps = frame_number / elapsed if elapsed > 0 else 0.0

        frame = draw_info(frame, last_count, fps, last_ms)

        cv2.imshow("AI VISION - Laptop Camera", frame)

        # waitKey(1) waits 1 millisecond and returns the key pressed.
        if cv2.waitKey(1) & 0xFF == ord("Q"):
            print("Stopping camera.")
            break

    camera.release()
    cv2.destroyAllWindows()

    print("")
    print("Camera mode finished after", frame_number, "frames.")


# ===========================================================================
# MODE 2: VIDEO FILE
# ===========================================================================

def run_video(model, device):
    """Detect on a video file, frame by frame, using stream=True."""
    print("")
    print("=" * 60)
    print("MODE 2 - VIDEO FILE")
    print("=" * 60)

    video_path = input("Enter the video file path: ").strip()

    if video_path == "":
        print("No path given, going back to the menu.")
        return

    if not os.path.exists(video_path):
        print("ERROR: file not found:", video_path)
        return

    print("")
    print("Press Q in the video window to stop early.")
    print("")

    # stream=True keeps results out of memory, which matters for long videos.
    results = model.predict(
        source=video_path,
        stream=True,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        device=device,
        verbose=False,
    )

    frame_number = 0
    total_objects = 0
    start_time = time.time()

    for result in results:
        frame = result.orig_img.copy()
        count = 0

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                coordinates = box.xyxy[0].tolist()
                class_name = result.names[class_id]

                draw_box(frame, coordinates, class_name, score)
                count += 1

        total_objects += count
        frame_number += 1

        inference_ms = 0.0
        if hasattr(result, "speed") and result.speed is not None:
            inference_ms = result.speed.get("inference", 0.0)

        elapsed = time.time() - start_time
        fps = frame_number / elapsed if elapsed > 0 else 0.0

        frame = draw_info(frame, count, fps, inference_ms)

        cv2.imshow("AI VISION - Video File", frame)

        if cv2.waitKey(1) & 0xFF == ord("Q"):
            print("Stopped early.")
            break

    cv2.destroyAllWindows()

    elapsed_total = time.time() - start_time
    average_fps = frame_number / elapsed_total if elapsed_total > 0 else 0.0

    print("")
    print("=" * 60)
    print("VIDEO DONE")
    print("=" * 60)
    print("  frames processed :", frame_number)
    print("  average FPS      :", round(average_fps, 2))
    print("  total time       :", round(elapsed_total, 1), "seconds")
    print("  objects detected :", total_objects)
    print("=" * 60)

    save_path = ask_save_path("output_detection.mp4", [".mp4"])

    if save_path:
        save_processed_video(video_path, save_path, model, device)


def save_processed_video(video_path, save_path, model, device):
    """Re-run the video and write the drawn frames to a new file."""
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print("ERROR: cannot open the video for saving.")
        return

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30.0

    # mp4v works for .mp4 output files.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("ERROR: cannot create the output file.")
        capture.release()
        return

    print("Saving to", save_path, "...")

    results = model.predict(
        source=video_path,
        stream=True,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        device=device,
        verbose=False,
    )

    saved_frames = 0

    for result in results:
        frame = result.orig_img.copy()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                coordinates = box.xyxy[0].tolist()
                class_name = result.names[class_id]

                draw_box(frame, coordinates, class_name, score)

        writer.write(frame)
        saved_frames += 1

    capture.release()
    writer.release()

    print("Saved", saved_frames, "frames to", save_path)


# ===========================================================================
# MODE 3: IMAGE FILE
# ===========================================================================

def run_image(model, device):
    """Detect on one picture and show the result."""
    print("")
    print("=" * 60)
    print("MODE 3 - IMAGE FILE")
    print("=" * 60)

    image_path = input("Enter the image file path: ").strip()

    if image_path == "":
        print("No path given, going back to the menu.")
        return

    if not os.path.exists(image_path):
        print("ERROR: file not found:", image_path)
        return

    frame = cv2.imread(image_path)

    if frame is None:
        print("ERROR: cannot read the image file.")
        return

    # An image has no frame rate, so show a placeholder FPS value.
    frame, count, inference_ms = read_one_frame(model, device, frame)
    frame = draw_info(frame, count, 0.0, inference_ms)

    print("")
    print("Detected objects:", count)
    print("Inference time  :", round(inference_ms, 1), "ms")

    cv2.imshow("AI VISION - Image File", frame)
    print("")
    print("Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_path = ask_save_path("detected_image.jpg", [".jpg", ".jpeg", ".png"])

    if save_path:
        cv2.imwrite(save_path, frame)
        print("Saved result to", save_path)


# ===========================================================================
# MENU
# ===========================================================================

def show_menu():
    """Print the menu and return the user's choice."""
    print("")
    print("=" * 60)
    print("AI-BASED UNDERWATER VISUAL PERCEPTION")
    print("Real-Time ROV Object Detection")
    print("=" * 60)
    print("")
    print("  1. Laptop Camera")
    print("  2. Video File")
    print("  3. Image File")
    print("  0. Exit")
    print("")
    choice = input("Enter your choice (1, 2, 3 or 0): ").strip()
    return choice


def main():
    print("=" * 60)
    print("AI VISION APPLICATION")
    print("=" * 60)
    print("  model      :", MODEL_PATH)
    print("  confidence :", CONFIDENCE)
    print("  image size :", IMGSZ)
    print("  device set :", DEVICE)

    # Work out which device to actually use.
    if DEVICE == "auto":
        device = choose_device()
    else:
        device = DEVICE

    print("  device used:", device)
    print("=" * 60)

    model = load_model(device)

    if model is None:
        return

    # Keep showing the menu until the user chooses to exit.
    while True:
        choice = show_menu()

        if choice == "1":
            run_camera(model, device)

        elif choice == "2":
            run_video(model, device)

        elif choice == "3":
            run_image(model, device)

        elif choice == "0":
            print("")
            print("Closing the application.")
            break

        else:
            print("")
            print("Please enter 1, 2, 3 or 0.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
