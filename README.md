# Speed Sign Board Detection and Speed Control System

## Overview
The project utilizes Deep Learning to detect speed signboards from a live stream. Based on the detected sign, it controls a motor and influences the movement of a car. It integrates computer vision with robotics to enable semi-autonomous behaviour based on environmental cues.

## Key Features

## Hardware Components

- **Raspberry Pi 5** - The Main Processing Unit
- **Camera Module**  - To Capture real-time video for signboard detection.
- **LN298N Motor Driver and 4BO Motors** - To control the car's movement
- **Battery Powered Power Supply** - To Power the Raspberry Pi and Motors

## Software Components

- **Python** - Used for implementing Deep Learning Logic and Motor Control Logic
- **YOLO** - A Real-time object detection system used for identifying signboards. We utilize Ultralytics YOLOv8 implementation of YOLO for inference.
- **OpenCV** - used for video processing tasks, for reading from the live camera, pre-processing frames for the YOLO Model, and potentially draw bounding boxes around the detected signs.
- **gpiozero** - used for interfacing with Raspberry Pi's GPIO Pin, which is used to control the car's motors.
## Installation Steps

- Connect the GPIO Pins to the Motor Driver in such a way that:
  - GPIO12 (PIN 12) to ENA 
  - GPIO21 (PIN 40) to ENB 
  - GPIO16 (PIN 36) to IN1
  - GPIO12 (PIN 32) to IN2
  - GPIO 6 (PIN 31) to IN3
  - GPIO 5 (PIN 29) to IN4
  - GND        (PIN39)  to GND
    
## Usage 

Directions to run the code
1. Unzip the whole repository and make it your current directory 
2. Install all the required dependencies using the requirments.txt file
    * Open the Terminal in Raspberry Pi OS
    * **Type** - `pip install -r requirements.txt`
3. For running on USB Camera 
    * **Type** - `python detect.py Models/speed_sign_board.pt --source usb0`
    * Or if you're using Picamera
    * **Type** - `python detect.py --model Models/speed_sign_board.pt --source picamera0 --resolution 640x840`
4. For Motor Control
   * Open a new window in terminal
   * **Type** - `python control.py`
  

