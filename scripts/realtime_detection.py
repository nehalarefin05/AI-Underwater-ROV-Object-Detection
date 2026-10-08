"""
realtime_detection.py

Real-time underwater debris detection on a video, for an ROV camera feed.

It reads a video frame by frame, runs YOLO detection on each frame, draws
the bounding boxes with the class name and confidence, shows the live FPS
and the inference time, and saves the result as a new .mp4 file.

Run:
    python scripts/realtime_detection.py
"""

# ---------------------------------------------------------------------------
# SETTINGS  (change these only)
# ---------------------------------------------------------------------------

MODEL_PATH = "runs/detect/runs/yolo26n_baseline/weights/best.pt"

VIDEO_PATH = "input_video.mp4"        # the video you want to process
OUTPUT_PATH = "output_detection.mp4"  # where the result is saved

IMGSZ = 640        # detection image size
DEVICE = 0         # 0 = use the first GPU, "cpu" = use the CPU
CONFIDENCE = 0.25  # only show boxes above this score (0 to 1)

# Colours used for the boxes: BGR format (OpenCV order)
BOX_COLOUR = (0, 255, 0)   # green
TEXT_COLOUR = (255, 255, 255)

FONT = 0                # OpenCV font. 0 = FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6         # text size
FONT_THICKNESS = 2       # text thickness

# ---------------------------------------------------------------------------

import os
import time

import cv2
from ultralytics import YOLO


def draw_detection(box, class_name, score, frame):
    """Draw one bounding box, its class name and its confidence."""
    # The box comes as [left, top, right, bottom].
    left = int(box[0])
    top = int(box[1])
    right = int(box[2])
    bottom = int(box[3])

    cv2.rectangle(frame, (left, top), (right, bottom), BOX_COLOUR, 2)

    label = class_name + " " + str(round(float(score), 2))

    cv2.putText(frame, label, (left, top - 8),
                FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICKNESS)


def draw_status(frame, fps, inference_ms, total_detections):
    """
    Draw the status panel in the top-left corner:
      AI VISION: ACTIVE
      FPS, inference time, detection count
    """
    panel_height = 78
    panel_width = 290

    # A filled dark box makes the text easy to read over any video.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, "AI VISION: ACTIVE", (12, 26),
                FONT, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "FPS: " + str(round(fps, 1)), (12, 50),
                FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICKNESS)
    cv2.putText(frame, "Inference: " + str(round(inference_ms, 1)) + " ms",
                (12, 72), FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICKNESS)
    cv2.putText(frame, "Objects: " + str(total_detections), (12, 94),
                FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICKNESS)

    return frame


def process_video():
    """Read the video, detect debris in every frame, and save the result."""

    # ------------------------------------------------------------------
    # 1. Check everything is in place before starting.
    # ------------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print("ERROR: model file not found:", MODEL_PATH)
        print("Train the model first with:  python scripts/train.py")
        return

    if not os.path.exists(VIDEO_PATH):
        print("ERROR: video file not found:", VIDEO_PATH)
        print("Set VIDEO_PATH at the top of this script.")
        return

    print("Loading model :", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    print("Opening video :", VIDEO_PATH)
    capture = cv2.VideoCapture(VIDEO_PATH)

    if not capture.isOpened():
        print("ERROR: cannot open the video file.")
        return

    # ------------------------------------------------------------------
    # 2. Get the video size and frame rate.
    # ------------------------------------------------------------------
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    # Some videos report a frame rate of 0, so use a sensible fallback.
    if fps is None or fps <= 0:
        fps = 30.0
        print("Note: video did not report an FPS, using 30.")

    print("Video size    :", width, "x", height)
    print("Video FPS     :", round(fps, 2))

    # ------------------------------------------------------------------
    # 3. Open the output video writer.
    #    mp4v is a safe four-character code for .mp4 files.
    # ------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("ERROR: cannot create the output file:", OUTPUT_PATH)
        capture.release()
        return

    print("Saving to     :", OUTPUT_PATH)
    print("")
    print("Press 'q' to stop early.")
    print("")

    # ------------------------------------------------------------------
    # 4. Frame counters for the live FPS.
    # ------------------------------------------------------------------
    frame_count = 0
    total_detections = 0
    start_time = time.time()

    # stream=True means results are not stored in memory. This keeps
    # memory usage low, which matters for a long ROV video.
    results = model.predict(
        source=VIDEO_PATH,
        stream=True,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # 5. Main loop: one frame at a time.
    # ------------------------------------------------------------------
    for result in results:

        # Ultralytics gives us the frame with boxes already drawn.
        # We draw our own text on top instead, so the style is ours.
        frame = result.orig_img.copy()

        frame_detections = 0

        if result.boxes is not None:
            for box in result.boxes:

                class_id = int(box.cls[0])
                score = float(box.conf[0])
                coordinates = box.xyxy[0].tolist()

                # The model knows the class names from the dataset YAML.
                class_name = result.names[class_id]

                draw_detection(coordinates, class_name, score, frame)
                frame_detections += 1

        total_detections += frame_detections

        # Timing for this frame.
        inference_ms = 0.0
        if hasattr(result, "speed") and result.speed is not None:
            inference_ms = result.speed.get("inference", 0.0)

        frame_count += 1
        elapsed = time.time() - start_time
        live_fps = frame_count / elapsed if elapsed > 0 else 0.0

        frame = draw_status(frame, live_fps, inference_ms, total_detections)

        writer.write(frame)

        # Show the frame on screen.
        cv2.imshow("AI VISION - Real-Time ROV Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

    # ------------------------------------------------------------------
    # 6. Clean up.
    # ------------------------------------------------------------------
    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed_total = time.time() - start_time
    average_fps = frame_count / elapsed_total if elapsed_total > 0 else 0.0

    print("")
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print("  frames processed :", frame_count)
    print("  average FPS      :", round(average_fps, 2))
    print("  total time       :", round(elapsed_total, 1), "seconds")
    print("  detections made  :", total_detections)
    print("  saved video      :", OUTPUT_PATH)
    print("=" * 60)


def main():
    print("=" * 60)
    print("REAL-TIME UNDERWATER DEBRIS DETECTION")
    print("=" * 60)
    print("  model       :", MODEL_PATH)
    print("  input video :", VIDEO_PATH)
    print("  output video:", OUTPUT_PATH)
    print("  image size  :", IMGSZ)
    print("  device      :", DEVICE)
    print("  confidence  :", CONFIDENCE)
    print("=" * 60)
    print("")

    process_video()


if __name__ == "__main__":
    main()
