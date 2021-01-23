##################################################################################################################

# This module defines all the functions for image processing in order to detect lines and calculate steering angle.

###################################################################################################################

# Import packages
import cv2
import numpy as np
import math


'''
description: Calculate region of interest 
param: image, points
return: image
'''
def ROI(img, vertices):
    mask = np.zeros_like(img)
    matchMaskColor = 255
    cv2.fillPoly(mask, vertices, matchMaskColor)
    maskedImage = cv2.bitwise_and(img, mask)
    return maskedImage


'''
description: Display detected lines in a separate window 
param: image, lines
return: image
'''
def displayLines(img, lines):
    copyImage = np.copy(img)
    emptyImage = np.zeros((copyImage.shape[0], copyImage.shape[1], 3), dtype=np.uint8)

    if lines is None:
        return None
    else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(emptyImage, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
        img = cv2.addWeighted(img, 0.8, emptyImage, 1, 0.0)

    return img


'''
description: Calculate ceneter lines from left and right boundry lines 
param: image, lines
return: lines
'''
def findBoundryLines(img, lines):
    height, width, channel = img.shape
    laneLines = []
    leftLaneLine = []
    rightLaneLine = []
    leftBoundary = width * 2/3
    rightBoundary = width / 3

    if lines is None:
        print('No lane lines detected')
        return laneLines
    else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < leftBoundary and x2 < leftBoundary:
                        leftLaneLine.append((slope, intercept))
                else:
                    if x1 > rightBoundary and x2 > rightBoundary:
                        rightLaneLine.append((slope, intercept))

        leftAverage = np.average(leftLaneLine, axis=0)
        if len(leftLaneLine) > 0:
            slope, intercept = leftAverage
            y1 = height  # bottom of the frame
            y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

            # bound the coordinates within the frame
            x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
            x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
            laneLines.append([[x1, y1, x2, y2]])
        rightAverage = np.average(rightLaneLine, axis=0)

        if len(rightLaneLine) > 0:
            slope, intercept = rightAverage
            y1 = height  # bottom of the frame
            y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

            # bound the coordinates within the frame
            x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
            x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
            laneLines.append([[x1, y1, x2, y2]])
            print(len(laneLines))

    return laneLines


'''
description: Calculate steering angle from the center line(caluculated above) 
param: image, lines
return: int
'''
def findSteeringAngle(frame, laneLines):
    if laneLines == 0:
        return -90
    else:
        height, width, _ = frame.shape
        if len(laneLines) == 1:
            x1, _, x2, _ = laneLines[0][0]
            x_offset = x2 - x1
        else:
            _, _, left_x2, _ = laneLines[0][0]
            _, _, right_x2, _ = laneLines[1][0]
            camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
            mid = int(width / 2 * (1 + camera_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid

        # find the steering angle, which is angle between navigation direction to end of center line
        y_offset = int(height / 2)

        angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
        steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

        return steering_angle


'''
description: Core function of the module
             - returns steering angle when you pass images as a parameter 
             - call all the function we defined above 
param: image
return: int(steering angle)
'''
def steeringAngleFromImage(image):
    i=0

    #------------------------------   Image Preprocessing steps   --------------------------------------------

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    dialation = cv2.dilate(image, kernel, iterations=1)
    blackwhiteimage = cv2.cvtColor(dialation, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(blackwhiteimage, 100, 200, apertureSize=3)


    ##----------------------------------   Display Images from each preprocessing step  ----------------------------
    cv2.imshow('Original', image)
    cv2.imshow("dialation", dialation)
    cv2.imshow('B/W', blackwhiteimage)
    cv2.imshow("erosion", erosion)
    cv2.imshow('Edge Detection', edges)


    #---------------------------------------  Define the Region of Interest    -------------------------------------

    height = edges.shape[0]
    width = edges.shape[1]

    ROI_vertices = [
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height)
    ]

    ROIimage = ROI(edges, np.array([ROI_vertices], np.int32), )

    #Detecting Lines using HoughLines Algorithm
    lines = cv2.HoughLinesP(ROIimage, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    lane_lines = findBoundryLines(image, lines)
    steeringAngle = findSteeringAngle(image, lane_lines)
    return steeringAngle

'''
description: main function - testing all the functions we have defined in the module 
param: None
return: none
'''
def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3, 640)
    camera.set(4, 480)

    while (camera.isOpened()):

        ret, image = camera.read()
        steeringAngle = steeringAngleFromImage(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(steeringAngle), (0, 130), font, 1, (200, 255, 255), 2, cv2.LINE_AA)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


## Calling the main function

if __name__ == '__main__':
    main()
