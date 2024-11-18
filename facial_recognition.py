import pickle
import time

import cv2
import face_recognition
import numpy as np

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the USB camera (change the camera index if needed)
camera_index = 0  # Default camera index, change to 1 or higher for other USB cameras
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize our variables
cv_scaler = 4  # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0


def process_frame(frame):
    global face_locations, face_encodings, face_names

    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))

    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame, face_locations, model="large"
    )

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    return frame


def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)

        # Draw a label with a name below the face
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


while True:
    # Capture a frame from the USB camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Process the frame with the function
    processed_frame = process_frame(frame)

    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)

    # Calculate and update FPS
    current_fps = calculate_fps()

    # Attach FPS counter to the text and boxes
    cv2.putText(
        display_frame,
        f"FPS: {current_fps:.1f}",
        (display_frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Display everything over the video feed.
    cv2.imshow("Video", display_frame)

    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cap.release()
cv2.destroyAllWindows()