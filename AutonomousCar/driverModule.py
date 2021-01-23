##################################################################################################################

# This module initializes hardware parts and control the motion(speed and steering wheels) of the car

###################################################################################################################

# Import Packages
import logging
import picar
import cv2
from Modules.laneDetectionModule import *


class RoboCar(object):

    __INITIAL_SPEED = 0
    __INITIAL_STEERING_ANGLE = 95

    def __init__(self):
        """ Initualization of all servos - camera calibration and front and back wheels"""

        logging.info('RoboCar getting ready...')

        picar.setup()

        self.camera = cv2.VideoCapture(-1)

        logging.debug('Init Camera Calibration')
        self.panServo = picar.Servo.Servo(1)
        self.panServo.write(90)

        self.tiltServo = picar.Servo.Servo(2)
        self.tiltServo.write(90)

        logging.debug('Setting up back wheels')
        self.backWheels = picar.back_wheels.Back_Wheels()
        self.backWheels.speed = 0

        ''' 
        Note: IN MY CASE, the center angle of steering wheel is 95, however normaly it is 90 degree
              Set up self.front_wheels.turn(angle) according your servo.
        '''
        logging.debug('Setting up front wheels')
        self.frontWheels = picar.front_wheels.Front_Wheels()
        self.frontWheels.turn(95)  # Steering Range is 45 - 135, for me the ceneter position is at angle 95

        logging.info('RoboCar is ready')

    def resetCar(self):
        
        """ Resetting the hardware modules"""
        logging.info('resetting the hardware modules.')
        
        self.backWheels.speed = 0
        self.frontWheels.turn(90)
        self.camera.release()
        cv2.destroyAllWindows()

    def drive(self, steeringAngle = __INITIAL_STEERING_ANGLE, speed=__INITIAL_SPEED):
        '''
        Note: Put the car in drive mode
        '''
        i = 0

        logging.info('The Car Speed is  set to %s...' % speed)
        self.backWheels.speed = speed

        logging.info('The steering angle is set to   %s...' % speed)
        self.frontWheels.turn = steeringAngle

        while self.camera.isOpened():
            ret , image = self.camera.read()
            calculateSteeringAngle = steeringAngleFromImage(image)
            self.frontWheels.turn = calculateSteeringAngle

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                break


def main():
    with RoboCar() as car:
        car.drive(95, 30)

if __name__ == '__main__':
    main()
