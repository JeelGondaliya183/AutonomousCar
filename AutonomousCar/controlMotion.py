import logging
import picar
import cv2
import numpy as np


def carSetUp():
    picar.setup()

    panServo = picar.Servo.Servo(1)
    panServo.write(60)

    tiltServo = picar.Servo.Servo(2)
    tiltServo.write(60)

    logging.debug('Set up back wheels')
    backWheels = picar.back_wheels.Back_Wheels()
    backWheels.speed = 0  # Speed Range is 0 (stop) - 100 (fastest)

    logging.debug('Set up front wheels')
    speedServo = picar.Servo.Servo(0)
    speedServo.write(60)


def controlSteeringAngle(angle):
    logging.debug('Set up front wheels')
    speedServo = picar.Servo.Servo(0)
    speedServo.write(angle)


def main():
    carSetUp()
    steeringAngle = 60
    camera = cv2.VideoCapture(0)
    # camera.set(3, 640)
    # camera.set(4, 480)
    i = 0
    while (camera.isOpened()):

        # original images from camera
        ret, image = camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        copy_image = np.copy(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(steeringAngle), (0, 130), font, 1, (200, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Original', image)
        cv2.imwrite("%03d_%03d.png" % (i, steeringAngle), copy_image)
        k = cv2.waitKey(1)
        if k == ord('d'):
            steeringAngle = steeringAngle + 20
            controlSteeringAngle(steeringAngle)

        elif k == ord('a'):
            steeringAngle = steeringAngle - 20
            controlSteeringAngle(steeringAngle)

        elif k == ord('s'):
            steeringAngle = steeringAngle - 20
            controlSteeringAngle(steeringAngle)
        elif k == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
