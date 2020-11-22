import cv2
import numpy as np
if __name__ == '__main__':
    cap = cv2.VideoCapture('vid.mp4')
    while True:
        _, img = cap.read() # GET THE IMAGE
        #img = cv2.resize(img,(640,480)) # RESIZE
        cv2.show(img)
        cv2.waitKey(1)


