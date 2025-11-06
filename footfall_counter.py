from google.colab import drive
drive.mount('/content/drive')


!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python-headless

!git clone https://github.com/ultralytics/yolov5
!pip install -r requirements.txt

!git clone https://github.com/abewley/sort.git
import sys
sys.path.append('/content/sort')

!pip install filterpy

import torch
import cv2
from sort.sort import Sort
from google.colab.patches import cv2_imshow
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

video_path = '/content/drive/MyDrive/Football Project/footfall_test.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read video")
frame_height, frame_width = frame.shape[:2]

roi_y_position = 800

entry_count = 0
exit_count = 0
memory = {}

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = '/content/drive/MyDrive/Football Project/footfall_output.mp4'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.pred[0].cpu().numpy()

    person_dets = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0 and conf > 0.5:
            person_dets.append([x1, y1, x2, y2, conf])

    dets = np.array(person_dets) if person_dets else np.empty((0,5))


    tracks = tracker.update(dets)

    current_memory = {}


    cv2.line(frame, (0, roi_y_position), (frame_width, roi_y_position), (0, 255, 255), 3)


    for *bbox, tid in tracks:
        x1, y1, x2, y2 = map(int, bbox)
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        current_memory[tid] = cy

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {int(tid)}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

        if tid in memory:
            prev_y = memory[tid]
            if prev_y < roi_y_position and cy >= roi_y_position:
                exit_count += 1
            elif prev_y > roi_y_position and cy <= roi_y_position:
                entry_count += 1

    memory = current_memory.copy()


    cv2.putText(frame, f'Entries: {entry_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    cv2.putText(frame, f'Exits: {exit_count}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2_imshow(frame)

    out.write(frame)

cap.release()
out.release()

