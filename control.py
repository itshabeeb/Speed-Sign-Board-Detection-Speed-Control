import RPi.GPIO as GPIO
import time

# Set up GPIO pins
ENA = 18  # PWM pin for Motor A
IN1 = 23  # Direction pin for Motor A
IN2 = 24  # Direction pin for Motor A

GPIO.setmode(GPIO.BCM)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# Set up PWM
pwm = GPIO.PWM(ENA, 1000)  # 1 kHz frequency
pwm.start(0)  # Start with 0% duty cycle

def set_motor_speed(speed):
    if speed == 20:
        duty_cycle = 40  # 40% speed
    elif speed == 40:
        duty_cycle = 70  # 70% speed
    elif speed == 60:
        duty_cycle = 100  # 100% speed
    else:
        duty_cycle = 0  # Stop motor if unknown speed

    # Set motor direction (forward example)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

    # Adjust speed using PWM
    pwm.ChangeDutyCycle(duty_cycle)

try:
    # Example: Respond to detected speeds
    while True:
        detected_speed = int(input("Enter detected speed (20, 40, 60): "))
        set_motor_speed(detected_speed)
        time.sleep(1)  # Adjust as needed
except KeyboardInterrupt:
    print("Stopping motors...")
finally:
    pwm.stop()
    GPIO.cleanup()
