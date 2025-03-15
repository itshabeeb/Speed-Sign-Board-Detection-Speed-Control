import cv2
import torch
import time
import numpy as np
import RPi.GPIO as GPIO

# Load trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.onnx')

# Setup Raspberry Pi GPIO
STOP_SIGNAL_PIN = 21  # GPIO pin to send signal
GPIO.setmode(GPIO.BCM)
GPIO.setup(STOP_SIGNAL_PIN, GPIO.OUT)

# Start camera capture
cap = cv2.VideoCapture(0)  # Adjust based on your camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Extract detected classes
    detected_class = None
    for *box, conf, cls in results.xyxy[0]:  
        detected_class = int(cls)  

    # Map class index to speed values (ensure this matches your training labels)
    class_to_speed = {0: 20, 1: 40, 2: 60, 3: "STOP"}  
    if detected_class in class_to_speed:
        detected_speed = class_to_speed[detected_class]
        print(f"Detected sign: {detected_speed}")

        # Send detected speed to `control.py` via GPIO
        if detected_speed == "STOP":
            GPIO.output(STOP_SIGNAL_PIN, GPIO.HIGH)  # Send stop signal
        else:
            GPIO.output(STOP_SIGNAL_PIN, GPIO.LOW)  # Clear stop signal

        # Save detected speed to a file for `control.py` to read
        with open("speed_signal.txt", "w") as f:
            f.write(str(detected_speed))

    time.sleep(1)  # Adjust as needed

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
