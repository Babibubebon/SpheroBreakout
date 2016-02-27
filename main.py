#!/bin/env python3
import cv2
import numpy as np

def main():
    capture = cv2.VideoCapture(0)
    
    if capture.isOpened() is False:
        raise("IO Error")
        
    # setting
    capture.set(cv2.CAP_PROP_FPS, 12)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

    # background image
    ret, bg = capture.read()
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)

    idx = 0

    while True:
        ret, image = capture.read()

        if ret == False:
            continue
        
        # filter
        img = cv2.GaussianBlur(image, (5, 5), 0)
        cimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 背景差分
        dimg = diffImage(bg, cimg)
        dimg = cv2.medianBlur(dimg, 5)
        ret, dimg = cv2.threshold(dimg, 10, 255, cv2.THRESH_TOZERO)
        dimg = cv2.equalizeHist(dimg)
        
        # detect circles
        circles = cv2.HoughCircles(dimg, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=32, param2=40, minRadius=20, maxRadius=50)
        drawCircles(image, circles)
        
        keyCode = cv2.waitKey(33)
        if keyCode >= 0:
            print("keyCode: {}".format(keyCode))
            if keyCode == 27:  # Esc
                cv2.imwrite("image.png", image)
                break
            if keyCode == 32:  # space
                idx = (idx + 1) % 2
        
        # show image
        image = [image, dimg][idx]
        cv2.imshow("Capture", image)

    cv2.destroyAllWindows()


def diffImage(bg, img):
    diff = bg.astype(np.int16) - img.astype(np.int16)
    return abs(diff).astype(np.uint8)


def drawCircles(image, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles)
        
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0,255,0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0,0,255), 3)

if __name__=="__main__":
    main()
