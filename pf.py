import cv2
import numpy as np
import serial
import tensorflow as tf

class PersonFollower:
    def __init__(self, serial_port, baud_rate=9600):
        self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=1)

    def move_towards(self, target_position, frame):
        if target_position is None:
            self.send_command("STOP", 0, 0)
            return

        x, y = target_position
        frame_width = frame.shape[1]

        # Define zones
        zone_width = frame_width // 12
        left_zone_6 = 0
        left_zone_5 = left_zone_6 + zone_width
        left_zone_4 = left_zone_5 + zone_width
        left_zone_3 = left_zone_4 + zone_width
        left_zone_2 = left_zone_3 + zone_width
        left_zone_1 = left_zone_2 + zone_width
        right_zone_1 = left_zone_1 + zone_width
        right_zone_2 = right_zone_1 + zone_width
        right_zone_3 = right_zone_2 + zone_width
        right_zone_4 = right_zone_3 + zone_width
        right_zone_5 = right_zone_4 + zone_width
        right_zone_6 = right_zone_5 + zone_width

        # Determine movement based on zone
        if x < left_zone_1:
            if x < left_zone_6:
                self.send_command("SHARP_LEFT", -1, 1)
            elif x < left_zone_5:
                self.send_command("MODERATE_LEFT", -0.7, 1)
            elif x < left_zone_4:
                self.send_command("SLIGHT_LEFT", -0.5, 1)
            elif x < left_zone_3:
                self.send_command("SLIGHT_LEFT", -0.3, 1)
            elif x < left_zone_2:
                self.send_command("SLIGHT_LEFT", -0.2, 1)
            else:
                self.send_command("SLIGHT_LEFT", -0.1, 1)
        elif x > right_zone_1:
            if x > right_zone_6:
                self.send_command("SHARP_RIGHT", 1, -1)
            elif x > right_zone_5:
                self.send_command("MODERATE_RIGHT", 1, -0.7)
            elif x > right_zone_4:
                self.send_command("SLIGHT_RIGHT", 1, -0.5)
            elif x > right_zone_3:
                self.send_command("SLIGHT_RIGHT", 1, -0.3)
            elif x > right_zone_2:
                self.send_command("SLIGHT_RIGHT", 1, -0.2)
            else:
                self.send_command("SLIGHT_RIGHT", 1, -0.1)
        else:
            self.send_command("FORWARD", 1, 1)

    def send_command(self, command, left_speed=0, right_speed=0):
        data = f"{command},{left_speed},{right_speed}\n"
        self.serial_conn.write(data.encode())

    def draw_bounding_box(self, frame, box, is_target=False):
        x, y, w, h = box
        center_x = x + w // 2
        center_y = y + h // 2

        if is_target:
            # Draw transparent white bounding box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), -1)
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw the center point in red
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        else:
            # Draw standard bounding box in blue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return center_x, center_y

    def process_frame(self, frame, interpreter):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.float32(input_data) / 255.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

        target_position = None
        for i in range(len(scores)):
            if scores[i] > 0.6 and classes[i] == 1:  # Assuming 'person' class is indexed by 1
                ymin, xmin, ymax, xmax = boxes[i]
                x = int(xmin * frame.shape[1])
                y = int(ymin * frame.shape[0])
                w = int((xmax - xmin) * frame.shape[1])
                h = int((ymax - ymin) * frame.shape[0])
                center_x, center_y = self.draw_bounding_box(frame, (x, y, w, h), is_target=True)
                target_position = (center_x, center_y)

        self.move_towards(target_position, frame)
        return frame

def main():
    serial_port = "/dev/ttymxc3"  # Replace with actual serial port
    baud_rate = 115200  # Set the baud rate according to your setup

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="lite_model.tflite")
    interpreter.allocate_tensors()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    follower = PersonFollower(serial_port, baud_rate)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = follower.process_frame(frame_rgb, interpreter)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
