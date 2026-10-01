# AI-Based Underwater Visual Perception for Real-Time ROV Object Detection

## 1. Project Overview

This project uses a YOLO-based object detection system for underwater ROV imagery.
The objective is to detect relevant underwater objects and present detections to the operator.

## 2. Dataset

- Dataset: Seaclear Marine Debris Dataset
- Total images: 8,610
- Total annotation instances: 31,555
- Number of classes: 40
- Train images: 3,574
- Validation images: 854
- Test images: 741
- Cross-domain test images: 3,441
- Annotation converted from COCO to YOLO format
- Temporal block-based splitting was used
- Original dataset was preserved

## 3. Model

- Model: YOLO26n
- Training: Transfer learning from pretrained weights
- Image size: 640
- Batch size: 16
- Device: NVIDIA Tesla T4
- Training target: 50 epochs
- Actual training: 49 epochs
- Early stopping: Yes
- Best model: best.pt
- Best epoch: 29

## 4. Validation Results

- Precision: 46.4%
- Recall: 31.1%
- mAP50: 32.9%
- mAP50-95: 20.8%

## 5. Test Results

- Precision: 30.1%
- Recall: 24.2%
- mAP50: 23.1%
- mAP50-95: 14.6%

## 6. Cross-Domain Results

Held-out domain:
Marseille / SIP-E323CV

- Precision: 32.1%
- Recall: 29.3%
- mAP50: 26.9%
- mAP50-95: 17.8%

The cross-domain set contains only the classes present in the held-out subset and must be interpreted separately from the normal 40-class test result.

## 7. Real-Time Video Inference

- Input resolution: 1920 x 1080
- Input FPS: 29.97
- Total frames processed: 1,425
- Average inference pipeline FPS: 19.56
- Total processing time: 72.9 seconds
- Detections made: 18,043
- Output video: rov_realtime_detection_final.mp4

The measured 19.56 FPS is the processing speed achieved on the Tesla T4 for this pipeline. It is below the source video's 29.97 FPS, so the result should be described as near-real-time processing rather than 30 FPS real-time playback.

## 8. Important Limitations

1. The normal train/validation/test split contains known near-duplicate overlap across splits, so those metrics may be optimistic.
2. The cross-domain test provides a separate generalization measurement.
3. Several classes are highly imbalanced.
4. Some rare classes have insufficient validation/test examples.
5. Visual inspection showed false positives and missed detections in some underwater and above-water frames.
6. The video demonstration is an external underwater ROV-style video and is not DUBO's own live camera feed.
7. No object tracking was implemented; detections are frame-based.

## 9. Current Project Status

Dataset preparation: COMPLETE
YOLO training: COMPLETE
Validation: COMPLETE
Test evaluation: COMPLETE
Cross-domain evaluation: COMPLETE
Image prediction: COMPLETE
Video inference: COMPLETE
Real-time inference script: COMPLETE
DUBO live camera integration: NOT YET IMPLEMENTED

## 10. Main Trained Model

best.pt
