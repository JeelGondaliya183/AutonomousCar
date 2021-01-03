import cv2
import numpy as np
import math

def ROI(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drawLines(img, lines):
    copy_image = np.copy(img)
    empty_image = np.zeros((copy_image.shape[0], copy_image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(empty_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

    img = cv2.addWeighted(img, 0.8, empty_image, 1, 0.0)
    return img

def find_center_lines(img, lines):
    height, width, channel = img.shape
    lane_lines = []
    left_lane_line = []
    right_lanel_line = []

    left_boundary = width * 2/3
    right_boundary = width / 3

    if lines is None:
        print('No lane lines detected')
        return lane_lines

    else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_boundary and x2 < left_boundary:
                        left_lane_line.append((slope, intercept))
                else:
                    if x1 > right_boundary and x2 > right_boundary:
                        right_lanel_line.append((slope, intercept))

        left_fit_average = np.average(left_lane_line, axis=0)
        if len(left_lane_line) > 0:
            slope, intercept = left_fit_average
            y1 = height  # bottom of the frame
            y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

            # bound the coordinates within the frame
            x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
            x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
            lane_lines.append([[x1, y1, x2, y2]])


        right_fit_average = np.average(right_lanel_line, axis=0)
        if len(right_lanel_line) > 0:
            slope, intercept = right_fit_average
            y1 = height  # bottom of the frame
            y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

            # bound the coordinates within the frame
            x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
            x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
            lane_lines.append([[x1, y1, x2, y2]])


            return lane_lines


def compute_steering_angle(frame, lane_lines):
    if len(lane_lines) == 0:
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    return steering_angle





def main():
    camera = cv2.VideoCapture(1)
    #camera.set(3, 640)
    #camera.set(4, 480)

    while (camera.isOpened()):

        # original images from camera
        ret, image = camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Original', image)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(image, kernel, iterations=1)

        dialation = cv2.dilate(image, kernel, iterations=1)

        cv2.imshow("dialation", dialation)
        # covert into gray scale
        b_w_image = cv2.cvtColor(dialation, cv2.COLOR_BGR2GRAY)
        cv2.imshow('B/W', b_w_image)

        cv2.imshow("erosion", erosion)

        edges = cv2.Canny(b_w_image, 100, 200, apertureSize=3)

        cv2.imshow('Edge Detection', edges)


        height = edges.shape[0]
        width = edges.shape[1]

        ROI_vertices = [
            (0, height * 1 / 2),
            (width, height * 1 / 2),
            (width, height),
            (0, height)
        ]

        ROIimage = ROI(edges, np.array([ROI_vertices], np.int32), )

        lines = cv2.HoughLinesP(ROIimage, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        lane_lines = find_center_lines(image, lines)
        lines_image = drawLines(image, lane_lines)
        steering_angle = compute_steering_angle(image, lane_lines)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(lines_image, str(steering_angle), (0,130), font, 1, (200,255,255), 2, cv2.LINE_AA)
        cv2.imshow('Image with Line detecion', lines_image)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
