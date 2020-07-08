import cv2 as cv
import numpy as np
from enum import Enum

class Detective:
    def __init__(self,lowerb,upperb,cornernum,data):
        """
        Initialize the members' values...

        Arguments:
        ----------
            data (): Image or video raw data for shape and color detection...
            lowerb (): Lower boundary of the wanted color in BGR form
            upperb (): Upper boundary of wanted color in BGR form
            cornernum (int): for detecting the shape by it's corner number
        """
        self.lowerb = lowerb
        self.upperb = upperb
        self.hsv_colors = cv.cvtColor(np.uint8([[lowerb,upperb]]), cv.COLOR_BGR2HSV)
        self.cornernum = cornernum
        self.data = data
        self.hsv = cv.cvtColor(self.data, cv.COLOR_BGR2HSV)

    def get_x_y(self, verbose=False):
        """
        Returns the shape boundaries drawen on image

        Arguments:
        ----------
        verbose (bool): shows every result... frame by frame...
        """
        mask = cv.inRange(self.hsv, self.hsv_colors[0][0], self.hsv_colors[0][1])
        kernel = np.ones((10,10),np.uint8)
        mask = cv.erode(mask, kernel)
        contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        x,y="",""
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .1*cv.arcLength(cnt,True), True)
            area = cv.contourArea(approx)
            x,y = approx[0][0]
            #print(x,y)
            if area >= 90 and len(approx) == self.cornernum:
                cv.drawContours(self.data, [approx], -1, (0,0,0), 10)
        # Results showed in seperate Windows...
        if verbose:
            cv.namedWindow("mask", cv.WINDOW_NORMAL)
            cv.imshow("mask", mask)
            cv.namedWindow("image", cv.WINDOW_NORMAL)
            cv.imshow("image", self.data)
            cv.namedWindow("hsv", cv.WINDOW_NORMAL)
            cv.imshow("hsv", self.hsv)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return (self.data,x,y)
