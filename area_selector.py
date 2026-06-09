import cv2
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / 'video'
VIDEO_CANDIDATES = [
   
    VIDEO_DIR / 'Car4.mp4',
    VIDEO_DIR / 'Car3.mp4'
]

video_path = None
for candidate in VIDEO_CANDIDATES:
    if candidate.exists():
        video_path = str(candidate)
        break

if video_path is None:
    raise FileNotFoundError(
        'No video file found. Put a supported MP4 in the "video" folder and update the code if needed.'
    )

# পলিগনের পয়েন্টগুলো রাখার জন্য লিস্ট
parking_area = []

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        parking_area.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # ভুল হলে রাইট ক্লিক করে শেষ পয়েন্ট ডিলিট করতে পারবে
        if len(parking_area) > 0:
            parking_area.pop()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f'Cannot open video file: {video_path}')

cv2.namedWindow('Select Area')
cv2.setMouseCallback('Select Area', mouse_click)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # ফ্রেমের সাইজ একটু ছোট বা বড় করতে চাইলে (ঐচ্ছিক)
    frame = cv2.resize(frame, (1020, 500))

    # ডট এবং পলিগন ড্র করা
    for pt in parking_area:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
    if len(parking_area) > 1:
        cv2.polylines(frame, [np.array(parking_area, np.int32)], True, (0, 255, 0), 2)

    cv2.imshow('Select Area', frame)
    
    key = cv2.waitKey(1)
    if key == 27: # ESC চাপলে কোঅর্ডিনেট সেভ হয়ে বন্ধ হবে
        if len(parking_area) == 0:
            print('No points were selected. Exiting without saving.')
        else:
            with open('parking_coordinates.pkl', 'wb') as f:
                pickle.dump(parking_area, f)
            print(f'Saved {len(parking_area)} points to parking_coordinates.pkl')
        break

cap.release()
cv2.destroyAllWindows()
