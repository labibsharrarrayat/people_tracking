import JETSON.GPIO as GPIO
import time

ha_pin = 5
hd_pin = 6
va_pin = 13
vd_pin = 19

GPIO.setmode(GPIO.BOARD)
GPIO.setup(ha_pin, GPIO.OUT)
GPIO.setup(hd_pin, GPIO.OUT)
GPIO.setup(va_pin, GPIO.OUT)
GPIO.setup(vd_pin, GPIO.OUT)


GPIO.output(ha_pin, GPIO.HIGH)
GPIO.output(hd_pin, GPIO.HIGH)

GPIO.output(va_pin, GPIO.LOW)
GPIO.output(vd_pin, GPIO.LOW)

time.sleep(2)

GPIO.output(ha_pin, GPIO.LOW)
GPIO.output(hd_pin, GPIO.LOW)

GPIO.output(va_pin, GPIO.HIGH)
GPIO.output(vd_pin, GPIO.HIGH)

time.sleep(2)
