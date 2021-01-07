# Autonomous Car using Raspberry Pi

## Phase 1: Assemble Hardware Parts 

## Phase 2: Control the Car Remotely using Keyboard
The reason behind adding this module is to save data to train CNN model, which we is used in phase 4.
The following are the function associated with keyboard keys:
W : Move Forward 
A: Turn Left by 20 degree agnle
D: Turn Right by 20 degree angle
S: Stop the Car
Q: Turn off the camera
Up Arrow: Increase the speed
Down Arrow:Decrease the speed

## Phase 3: Lane Detection using OpenCV Image Analysis
### Lane Detection
Used Image Analysis techniques like grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Transform line detection to detect lane lines from the image.
### Calculate Steering Angle
Assuming the Camera is caliberated to center, compute steering angle based on the two detected lane lines.
1. calculate the two points of each line, which intersects with center(of hight) line of the frame
2. Find the center point(steering angle) of the line, which passes through teo point(from the step

## Phase 4: Training Data Collection
1. Gathered muliple images from video streaming and saved location of each image in CSV file
2. Calculated steering angle based on the number of time keyboard key was pressed.

## Phase 5: Robust Lane Detection using Convolutional Neural Network
