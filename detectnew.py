import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
from gpiozero import Motor, PWMOutputDevice
import os
import sys
# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (default: "0.5")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise match source resolution',
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

# GPIO setup for motor control using gpiozero
# Speed control (PWM) for left and right motors
ENA = PWMOutputDevice(18)  # Left motor speed control
ENB = PWMOutputDevice(19)  # Right motor speed control

# Direction control for left and right motors
left_motor = Motor(forward=23, backward=24)  # Left motors: IN1, IN2
right_motor = Motor(forward=27, backward=22)  # Right motors: IN3, IN4

# Set motors to default backward direction
def set_motor_speed(left_speed, right_speed):
    """
    Sets the speed and default direction (backward).
    """
    # Set motor speeds (0 to 1 scale)
    ENA.value = left_speed
    ENB.value = right_speed

    # Run motors in backward direction
    left_motor.backward()
    right_motor.backward()

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine source type
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

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

# Initialize frame source (camera, video, etc.)
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

# Define colors for bounding boxes
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106)]

# Frame rate buffer
frame_rate_buffer = []
fps_avg_len = 200

# Inference loop
try:
    while True:
        t_start = time.perf_counter()

        # Capture frame
        ret, frame = cap.read() if source_type in ['video', 'usb'] else (True, cap.capture_array())
        if not ret or frame is None:
            print("End of stream or camera disconnected.")
            break

        # Resize frame if necessary
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        for i in range(len(detections)):
            # Extract detection details
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()  # Bounding box
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            if conf > min_thresh:
                # Draw bounding box
                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                # Add label text
                label = f'{classname}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Control motor based on detection
                if classname == '20':
                    set_motor_speed(0.4, 0.4)  # Slow speed for both sides
                elif classname == '40':
                    set_motor_speed(0.7, 0.7)  # Medium speed
                elif classname == '60':
                    set_motor_speed(1.0, 1.0)  # Full speed
                else:
                    set_motor_speed(0.0, 0.0)  # Stop motors

        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate_calc = 1 / (t_stop - t_start)
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)

        # Add FPS overlay
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display the frame
        cv2.imshow("YOLO Detection Results", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
            break

finally:
    # Stop the motors
    set_motor_speed(0.0, 0.0)
    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    cv2.destroyAllWindows()
