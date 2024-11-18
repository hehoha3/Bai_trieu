import os
import time

import cv2

# Change this to the name of the person you're photographing
PERSON_NAME = "jaryd"
DATA_DIR = "data"
DATA_SIZE = 100


def create_folder(name):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    person_folder = os.path.join(DATA_DIR, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    return person_folder


def capture_photos(name):
    folder = create_folder(name)

    # Initialize the USB camera (using OpenCV)
    cap = cv2.VideoCapture(
        0
    )  # 0 for the default camera, change to 1, 2 if necessary for other USB cameras

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    photo_count = 0

    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        # Capture frame from USB camera
        ret, frame = cap.read()

        # If the frame was not successfully captured, skip to next frame
        if not ret:
            print("Error: Failed to capture image.")
            continue

        # Display the frame
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):  # Space key
            while photo_count < DATA_SIZE:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to capture image.")
                    continue

                cv2.imshow("Capture", frame)

                # Increment the photo count and save the image
                filename = f"{name}_{photo_count}.jpg"
                filepath = os.path.join(folder, filename)

                cv2.waitKey(25)

                cv2.imwrite(filepath, frame)
                print(f"Photo {photo_count} saved: {filepath}")

                photo_count += 1

                # Delay to allow some time between captures (optional)
                time.sleep(0.1)

            print(f"Captured {DATA_SIZE} photos for {name}.")
            break  # Exit the loop after capturing 100 photos

        elif key == ord("q"):  # Q key to quit without capturing
            break

    # Clean up
    cv2.destroyAllWindows()
    cap.release()  # Release the camera
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")


if __name__ == "__main__":
    capture_photos(PERSON_NAME)
