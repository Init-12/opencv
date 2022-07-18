from cv2 import cv2

import numpy as np
hsv_min = np.array((0, 160, 170), np.uint8)
hsv_max = np.array((208, 255, 255), np.uint8)


def cont():
    img = cv2.imread('2z.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 5)

    cv2.imshow("2z.jpg", img)
    cv2.waitKey(0)


def contt():
    img = cv2.imread('1z.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("img", img)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    cv2.imshow("shapes", img)

def conttt():
    f = '1z.jpg'
    img = cv2.imread(f)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        if len(cnt) > 33:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img, ellipse, (0, 0, 0), 5)


    cv2.imshow('contours', img)

    cv2.waitKey()
    cv2.destroyAllWindows()


conttt()
contt()
cont()