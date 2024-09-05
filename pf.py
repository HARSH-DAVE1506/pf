import cv2
import numpy as np
import serial
import time
import tensorflow as tf
import json

class Follower:
    def __init__(self, model_path, label_path, serial_port, baud_rate=115200):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.labels = self.load_labels(label_path)
        self.serial_port = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  # Give some time for the connection to establish

    def load_labels(self, path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}

    def draw_bounding_box(self, frame, box, label, is_target=False):
        x, y, w, h = box
        center_x = x + w // 2
        center_y = y + h // 2
        
        color = (255, 255, 255) if is_target else (0, 255, 0)  # White for the target person
        transparency = 0.7 if is_target else 0.5  # 70% transparency for the target person

        # Draw bounding box with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        frame = cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0)
        
        # Draw a center point in red
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Add label to the bounding box
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return center_x, center_y

    def move_towards(self, target_position, frame):
        if target_position is None:
            self.send_command(0, 0)  # Stop both wheels
            return

        frame_center_x = frame.shape[1] // 2
        center_x = target_position[0]

        # Calculate the error (distance from the center)
        error = center_x - frame_center_x

        # Define the maximum error (half the frame width)
        max_error = frame.shape[1] // 2

        # Calculate the turn rate (-0.5 to 0.5)
        turn_rate = max(min(error / max_error, 0.5), -0.5)

        # Base speed (adjust as needed)
        base_speed = 0.3

        # Calculate left and right wheel speeds
        left_speed = base_speed - turn_rate
        right_speed = base_speed + turn_rate

        # Clamp speeds to the allowed range (-0.5 to 0.5)
        left_speed = max(min(left_speed, 0.5), -0.5)
        right_speed = max(min(right_speed, 0.5), -0.5)

        self.send_command(left_speed, right_speed)

    def send_command(self, left_speed, right_speed):
        command = {
            "T": 1,
            "L": round(left_speed, 2),
            "R": round(right_speed, 2)
        }
        command_json = json.dumps(command)
        self.serial_port.write(command_json.encode('utf-8') + b'\n')
        response = self.serial_port.readline().decode('utf-8').strip()
        print(f"Sent command: {command_json}, Received response: {response}")

    def process_frame(self, frame, interpreter):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))
        
        # Check if the model expects UINT8 data type
        if input_details[0]['dtype'] == np.uint8:
            input_data = np.uint8(input_data)
        else:
            input_data = np.float32(input_data) / 255.0

        input_data = np.expand_dims(input_data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

        target_position = None
        for i in range(len(scores)):
            if scores[i] > 0.6:
                ymin, xmin, ymax, xmax = boxes[i]
                x = int(xmin * frame.shape[1])
                y = int(ymin * frame.shape[0])
                w = int((xmax - xmin) * frame.shape[1])
                h = int((ymax - ymin) * frame.shape[0])
                
                class_id = int(classes[i])
                label = f"{self.labels[class_id]}: {scores[i]:.2f}"
                print(f"Detected: {label}")
                
                center_x, center_y = self.draw_bounding_box(frame, (x, y, w, h), label, is_target=(class_id == 0))
                
                if class_id == 0:  # Assuming 'person' class is indexed by 0
                    target_position = (center_x, center_y)

        self.move_towards(target_position, frame)
        return frame

def main():
    model_path = "1.tflite"
    label_path = "label.txt"  # Path to your label file
    serial_port = '/dev/ttymxc2'  # Adjust based on your setup
    baud_rate = 115200  # Explicitly set baud rate
    video_source = 0  # Webcam source

    follower = Follower(model_path, label_path, serial_port, baud_rate)

    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = follower.process_frame(frame_rgb, follower.interpreter)

        #cv2.imshow("Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
