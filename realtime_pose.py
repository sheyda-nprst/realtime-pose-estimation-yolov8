"""Real-time human pose estimation from a webcam with YOLOv8n-pose.

Usage:
    python realtime_pose.py            # default webcam
    python realtime_pose.py --source 1 # another camera or a video file path
Press 'q' to quit.
"""
import argparse

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0", help="camera index or video path")
parser.add_argument("--model", default="yolov8n-pose.pt", help="pose model weights")
parser.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
args = parser.parse_args()

model = YOLO(args.model)  # nano model: small and fast enough for CPU
source = int(args.source) if args.source.isdigit() else args.source

for result in model(source=source, stream=True, conf=args.conf, verbose=False):
    cv2.imshow("YOLOv8 pose", result.plot())  # boxes + 17 COCO keypoints
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
