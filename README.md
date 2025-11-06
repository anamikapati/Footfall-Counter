# Footfall Counter using Computer Vision 

This project detects, tracks, and counts people entering or exiting a region of interest (ROI) using YOLOv5 and SORT tracking.

## Steps
1. Detect humans in a video stream using YOLOv5.
2. Track each person frame-by-frame with SORT.
3. Define a virtual ROI line.
4. Count entries and exits when people cross the line.

## Tools
- Python, OpenCV, PyTorch
- YOLOv5, SORT
- Google Colab (GPU)

## Output
Processed video with bounding boxes, IDs, and live `IN/OUT` counts.
[Output Video](https://drive.google.com/file/d/1jv9dJ5DLiOv7P5u7Cu2NTuKbSmebP9Uv/view?usp=drive_link)
