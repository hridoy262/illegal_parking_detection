from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
import easyocr

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

app = Flask(__name__)

# Initialize EasyOCR Reader (English)
reader = easyocr.Reader(['en'], gpu=True) # Set gpu=False if you don't have a GPU

# à¦®à¦¡à§‡à¦² à¦“ à¦•à§‹à¦…à¦°à§�à¦¡à¦¿à¦¨à§‡à¦Ÿ à¦²à§‹à¦¡
model = YOLO(str(BASE_DIR / 'yolov8n.pt'))
coordinates_path = BASE_DIR / 'parking_coordinates.pkl'
if not coordinates_path.exists():
    raise FileNotFoundError(
        'parking_coordinates.pkl not found. Run `python area_selector.py` and save the parking polygon first.'
    )

with open(coordinates_path, 'rb') as f:
    parking_area = pickle.load(f)

if not parking_area:
    raise ValueError('Loaded parking area is empty. Use area_selector.py to define the polygon.')

# Global stats tracking
stats = {
    "in_zone": 0,
    "violations": 0,
    "recent_plates": [],
    "parked_vehicles": []
}

latest_frame = None
(BASE_DIR / 'captures').mkdir(exist_ok=True)

def generate_frames():
    global stats, latest_frame
    cap = cv2.VideoCapture(video_path)
    tracked_plates = {} # Cache for OCR results: {id: "PLATE_TEXT"}
    tracked_vehicles = {}
    # Distance threshold in pixels (e.g., if a car is within 150px of a restricted point)
    DISTANCE_THRESHOLD = 150 
    vehicle_last_pos = {} # Track movement to ensure it's actually parked
    ILLEGAL_TIME_LIMIT = 5
    SPEEDUP_FACTOR = 2  # Process every 2nd frame to make playback faster

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        for _ in range(SPEEDUP_FACTOR - 1):
            cap.grab()

        frame = cv2.resize(frame, (1020, 500))

        cv2.polylines(frame, [np.array(parking_area, np.int32)], True, (0, 255, 255), 2)
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

        current_frame_vehicle_ids = []
        current_violations = 0
        current_in_zone = 0

        if results and len(results) > 0:
            # Class 2: car, 3: motorcycle, 5: bus, 7: truck
            # Class 11: stop sign (Example of a reference object)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = getattr(results[0].boxes, 'id', None)
            
            if ids is None:
                ids = np.zeros(len(boxes))
            else:
                ids = ids.cpu().numpy().astype(int)

            for box, object_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int(y2)

                # পলিগন এরিয়া চেক করা
                is_inside = cv2.pointPolygonTest(np.array(parking_area, np.int32), (cx, cy), False) >= 0
                
                if is_inside:
                    current_frame_vehicle_ids.append(object_id)
                    
                    # Check if vehicle has moved significantly
                    current_pos = (cx, cy)
                    prev_pos = vehicle_last_pos.get(object_id, current_pos)
                    distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
                    
                    # Reset timer if the vehicle is moving (not actually parked)
                    if distance > 10: # Threshold in pixels
                        tracked_vehicles[object_id] = time.time()
                    
                    vehicle_last_pos[object_id] = current_pos

                    if object_id not in tracked_vehicles:
                        tracked_vehicles[object_id] = time.time()

                    stay_duration = time.time() - tracked_vehicles[object_id]
                    current_in_zone += 1

                    if stay_duration > ILLEGAL_TIME_LIMIT:
                        current_violations += 1
                        
                        # Run OCR only once per vehicle to maintain performance
                        if object_id not in tracked_plates:
                            crop = frame[y1:y2, x1:x2]
                            ocr_results = reader.readtext(crop)
                            # Filter for strings that look like plate numbers (len > 2)
                            detected_text = "".join([res[1] for res in ocr_results if len(res[1]) > 2]).strip().upper()
                            plate_text = detected_text if detected_text else f"UNK-{object_id:03d}"
                            
                            tracked_plates[object_id] = plate_text
                            if plate_text not in stats["recent_plates"]:
                                stats["recent_plates"].insert(0, plate_text)
                                stats["recent_plates"] = stats["recent_plates"][:5]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        display_text = f"ILLEGAL: {tracked_plates.get(object_id, '')}"
                        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Update global stats
        stats["in_zone"] = current_in_zone
        stats["violations"] = current_violations
        # বর্তমানে জোনের ভেতরে থাকা গাড়িগুলোর লিস্ট আপডেট করা
        stats["parked_vehicles"] = [tracked_plates.get(vid, f"Scanning (ID:{vid})") for vid in current_frame_vehicle_ids]
        latest_frame = frame.copy()

        for vid in list(tracked_vehicles.keys()):
            if vid not in current_frame_vehicle_ids:
                tracked_vehicles.pop(vid, None)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    return jsonify(stats)

@app.route('/api/capture', methods=['POST'])
def capture():
    global latest_frame
    if latest_frame is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"violation_{timestamp}.jpg"
        save_path = BASE_DIR / 'captures' / filename
        cv2.imwrite(str(save_path), latest_frame)
        return jsonify({"status": "success", "message": f"Saved as {filename}"})
    return jsonify({"status": "error", "message": "Frame not available"}), 500


if __name__ == '__main__':
    app.run(debug=True)
