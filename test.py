import cv2 as cv
import numpy as np

class Shape:
    def __init__(self, tmin, tmax, amin, amax):
        """
        Args:
        =====
        tmin: Threshold min val
        tmax: Threshold min val
        amin: min val of the object's area
        amax: max val of the object's area
        """
        self.tmin = tmin
        self.tmax = tmax
        self.amin = amin
        self.amax = amax

class Detective:
    def __init__(self, shape, verbose=True):
        """
        Args:
        =====
        shape (Shape): determines the specifications of the object which will be searched
        """
        if type(shape) == Shape:
            self.shape = shape
        else:
            raise "unk. type shape"
        self.verbose = verbose

img = cv.imread("media/out3.png")
img = cv.resize(img,(720,405))
img = cv.GaussianBlur(img, (7,7), 1)
gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, t = cv.threshold(gry, 150, 255, cv.THRESH_BINARY_INV)
contours, _ = cv.findContours(t, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    approx = cv.approxPolyDP(cnt,.001*cv.arcLength(cnt, True), True)
    area = cv.contourArea(approx)
    print("area:", area)
    if area <= 100000:
        M = cv.moments(approx)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print("center:",(cx,cy))
        cv.drawContours(img,[approx],-1,(0,0,0),10)
cv.imshow("t", t)
cv.imshow("o", img)
cv.waitKey(0)
cv.destroyAllWindows()
#cv.waitKey(0)
#cv.destroyAllWindows()
