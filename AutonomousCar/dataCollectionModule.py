##################################################################################################################

# This module stores images and steering angle in CSV file. Which we will further use in CNN model.

###################################################################################################################


#Import Packages

import cv2
import sys
from Modules.laneDetectionModule import  *


def saveTrainingData():
    camera = cv2.VideoCapture(-1)
    camera.set(3, 640)
    camera.set(4, 480)

    while (camera.isOpened()):
        i=0
        ret, image = camera.read()
        steeringAngle = steeringAngleFromImage(image)
        cv2.imwrite("%03d_%03d.png" % (i, steeringAngle), image)
        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    saveTrainingData()
