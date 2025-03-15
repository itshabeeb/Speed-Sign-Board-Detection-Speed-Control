import RPi.GPIO as GPIO
import time

# Define GPIO pins
ENA = 18  # PWM pin for Motor
IN1 = 23  # Motor direction pin
IN2 = 24  # Motor direction pin
STOP_SIGNAL_PIN = 21  # Receives stop signal from `detect.py`

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(STOP_SIGNAL_PIN, GPIO.IN)  # Read STOP signal

# Initialize PWM for speed control
pwm = GPIO.PWM(ENA, 1000)
pwm.start(0)

# Speed levels
speed_map = {20: 30, 40: 60, 60: 100}  # PWM duty cycle for each speed

def set_motor_speed(speed):
    if speed == "STOP":
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        pwm.ChangeDutyCycle(0)
        print("[INFO] Stop signal received. Motor stopped.")
    else:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        pwm.ChangeDutyCycle(speed_map[speed])
        print(f"[INFO] Motor running at {speed_map[speed]}% speed.")

while True:
    # Check stop signal
    if GPIO.input(STOP_SIGNAL_PIN) == GPIO.HIGH:
        set_motor_speed("STOP")
    else:
        try:
            with open("speed_signal.txt", "r") as f:
                speed_value = f.read().strip()
                if speed_value.isdigit():
                    set_motor_speed(int(speed_value))
        except FileNotFoundError:
            print("[INFO] No speed signal detected. Motor running.")

    time.sleep(1)  # Adjust polling rate as needed
