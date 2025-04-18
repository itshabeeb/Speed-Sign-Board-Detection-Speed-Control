# Speed Sign Board Detection using YOLOv11 and Speed Control System

## Overview
Our project explores speed control in real-life scenarios such as hospital zones, school zones, and highways, where different speed limits are critical for safety. The goal is to build a system that can detect speed signboards and adjust a car’s speed accordingly—just like how vehicles should behave in the real world.

To achieve this, our team developed a custom YOLOv11 model, a type of deep learning-based object detection algorithm. This model is trained to recognize speed signs like 20, 40, 60, and STOP.

The entire dataset used for training was created from scratch by capturing photos of speed signboards in various settings. Each image was carefully annotated and augmented using Roboflow. The final model was fine-tuned from a version of YOLOv11 pre-trained on the COCO dataset, which gave it a strong foundation to learn from fewer custom images.

In the prototype setup, the trained model runs in a simulated environment representing real-world locations. When it detects a speed signboard, the system automatically sends signals to a motor, adjusting the speed or movement of a small car—either slowing it down, speeding it up, or stopping it based on the sign detected.

This project is a hands-on example of how computer vision and robotics can work together to create smart, responsive systems that react to their environment—paving the way for semi-autonomous vehicles and real-world safety applications.

## Key Features

## Hardware Components

- **Raspberry Pi 5** - The Main Processing Unit
- **Camera Module**  - To Capture real-time video for signboard detection.
- **LN298N Motor Driver and 4BO Motors** - To control the car's movement
- **Battery Powered Power Supply** - To Power the Raspberry Pi and Motors

## Software Components

- **Python** - Used for implementing Deep Learning Logic and Motor Control Logic
- **YOLOv11** - A Real-time object detection system used for identifying signboards. We utilize Ultralytics YOLOv8 implementation of YOLO for inference.
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
  

