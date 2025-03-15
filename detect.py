import os
import sys
import argparse

import cv2
import numpy as np
from ultralytics import YOLO
import RPi.GPIO as GPIO  # GPIO library for Raspberry Pi

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# GPIO setup for motor control
ENA = 18  # PWM pin for Motor A
IN1 = 23  # Direction pin for Motor A
IN2 = 24  # Direction pin for Motor A

GPIO.setmode(GPIO.BCM)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# Set up PWM for motor control
pwm = GPIO.PWM(ENA, 1000)  # 1 kHz PWM frequency
pwm.start(0)  # Initially stopped

def set_motor_speed(speed):
    """Set motor speed based on detected speed sign."""
    if speed == 20:
        duty_cycle = 40  # 40% speed
    elif speed == 40:
        duty_cycle = 70  # 70% speed
    elif speed == 60:
        duty_cycle = 100  # 100% speed
    else:
        duty_cycle = 0  # Stop motor for unknown/invalid speed

    # Set motor direction to forward
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

    # Set the PWM duty cycle for speed control
    pwm.ChangeDutyCycle(duty_cycle)

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Video capture setup
if source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Main loop
try:
    while True:
        ret, frame = cap.read() if source_type in ['video', 'usb'] else (True, cap.capture_array())
        if not ret or frame is None:
            print("End of stream or camera disconnected.")
            break

        # Resize frame
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Run YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        for i in range(len(detections)):
            # Extract class, bounding box, and confidence
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf > min_thresh:
                if classname == '20':
                    set_motor_speed(20)
                elif classname == '40':
                    set_motor_speed(40)
                elif classname == '60':
                    set_motor_speed(60)
                else:
                    set_motor_speed(0)

        cv2.imshow("YOLO Detection Results", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
finally:
    pwm.stop()
    GPIO.cleanup()
    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    cv2.destroyAllWindows()
