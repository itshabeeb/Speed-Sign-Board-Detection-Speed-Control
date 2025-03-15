import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO  # Import GPIO
from ultralytics import YOLO

# Define GPIO setup
STOP_SIGNAL_PIN = 21  # Choose an unused GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(STOP_SIGNAL_PIN, GPIO.OUT)

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
args = parser.parse_args()

# Load model
model_path = args.model
model = YOLO(model_path, task='detect')
labels = model.names  # Get class labels

# Parse source
img_source = args.source
cap = cv2.VideoCapture(img_source if img_source.isnumeric() else int(img_source))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_detected = False
    for i in range(len(detections)):
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        if classname.lower() == "stop sign":  # Detect specific object
            print("[INFO] Stop sign detected!")
            GPIO.output(STOP_SIGNAL_PIN, GPIO.HIGH)  # Send signal to control.py
            object_detected = True
            break  # Stop checking further detections

    if not object_detected:
        GPIO.output(STOP_SIGNAL_PIN, GPIO.LOW)  # Ensure signal is off

    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
