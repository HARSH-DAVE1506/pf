import cv2
import mediapipe as mp
import numpy as np
import serial  
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import json
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

serial_port = '/dev/ttymxc3'  
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Initialize the hand tracking model
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)

# Pan and tilt angles initialized
pan_angle = 0
tilt_angle = 0

# Frame dimensions
dispW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
dispH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the center of the frame
frame_center_x = dispW // 2  # Center in the X direction
frame_center_y = dispH // 2  # Center in the Y direction

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the hand center
            x_coords = [landmark.x * dispW for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * dispH for landmark in hand_landmarks.landmark]
            hand_center_x = int(sum(x_coords) / len(x_coords))
            hand_center_y = int(sum(y_coords) / len(y_coords))

            # Calculate error from the center of the frame
            error_x = hand_center_x - frame_center_x
            error_y = hand_center_y - frame_center_y

            # Pan logic
            if error_x > 30:  # Hand is to the right
                pan_angle -= 1
                if pan_angle < -180:
                    pan_angle = -180
            elif error_x < -30:  # Hand is to the left
                pan_angle += 1
                if pan_angle > 180:
                    pan_angle = 180

            # Tilt logic
            if error_y > 30:  # Hand is below the center
                tilt_angle += 1
                if tilt_angle > 90:
                    tilt_angle = 90
            elif error_y < -30:  # Hand is above the center
                tilt_angle -= 1
                if tilt_angle < -30:
                    tilt_angle = -30

            # Prepare command
            command = {
                "T": 133,
                "X": pan_angle,
                "Y": tilt_angle,
                "SPD": 0,
                "ACC": 0
            }

            json_command = json.dumps(command)
            print("Sending command:", json_command)

            try:
                ser.write((json_command + '\n').encode('utf-8'))
                print("Command sent successfully")
            except serial.SerialException as e:
                print(f'Failed to send command: {e}')

    # Display the frame
    #cv2.imshow('Hand Tracking', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
