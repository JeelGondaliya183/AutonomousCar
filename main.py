import cv2


def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3, 640)
    camera.set(4, 480)

    while (camera.isOpened()):
        _, image = camera.read()
        cv2.imshow('Original', image)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        cv2.imshow('B/W', hsv_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()