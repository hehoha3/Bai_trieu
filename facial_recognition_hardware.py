import pickle
import time

import cv2
import face_recognition
import numpy as np
import paho.mqtt.client as mqtt

MQTT_HOST = "192.168.1.29"
MQTT_POST = 1883
KEEP_ALIVE = 60
TOPIC_PUB = "relay"

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the USB Camera
print("[INFO] initializing USB camera...")
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


# Initialize variables
cv_scaler = 4  # Scale factor for resizing frames to improve performance
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0


def process_frame(frame):
    global face_locations, face_encodings, face_names

    # Resize the frame to improve performance
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))

    # Convert the image from BGR to RGB color space
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get face encodings
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame, face_locations, model="large"
    )

    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.45
        )
        name = "Unknown"

        # Use the known face with the smallest distance
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            client.publish(TOPIC_PUB, "Hello trieu")

        face_names.append(name)

    return frame


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(
            frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    return frame


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


client = mqtt.Client()
client.connect(MQTT_HOST, MQTT_POST, KEEP_ALIVE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to capture video")
        break

    processed_frame = process_frame(frame)
    display_frame = draw_results(processed_frame)
    current_fps = calculate_fps()

    cv2.putText(
        display_frame,
        f"FPS: {current_fps:.1f}",
        (display_frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Video", display_frame)

    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
