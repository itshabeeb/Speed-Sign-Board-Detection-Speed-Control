import RPi.GPIO as GPIO
import time

# Define GPIO pins
ENA = 18  # PWM pin for Motor A
IN1 = 23  # Direction pin for Motor A
IN2 = 24  # Direction pin for Motor A
STOP_SIGNAL_PIN = 21  # Must match detect.py

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(STOP_SIGNAL_PIN, GPIO.IN)  # Listen for stop signal

pwm = GPIO.PWM(ENA, 1000)  # 1kHz PWM frequency
pwm.start(50)  # Start motor at 50% speed

try:
    while True:
        stop_signal = GPIO.input(STOP_SIGNAL_PIN)  # Read signal from detect.py

        if stop_signal == GPIO.HIGH:
            print("[INFO] Stop signal received! Stopping motor.")
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.LOW)  # Stop motor
        else:
            print("[INFO] No stop signal. Motor running.")
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)  # Move forward

        time.sleep(0.1)  # Check every 100ms

except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
