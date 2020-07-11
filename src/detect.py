import cv2 as cv
import numpy as np
from enum import Enum

class Detective:
    def __init__(self,lowerb,upperb,cornernum,area,data):
        """
        Initialize the members' values...

        Arguments:
        ----------
            data (): Image or video raw data for shape and color detection...
            lowerb (): Lower boundary of the wanted color in BGR form
            upperb (): Upper boundary of wanted color in BGR form
            cornernum (int): for detecting the shape by it's corner number
            area (int): for detecting the shape by it's area
        """
        self.lowerb = lowerb
        self.upperb = upperb
        self.cornernum = cornernum
        self.area = area
        self.data = data
        self.hsv_colors = self.cnvt2HSV()
        self.hsv = self.calcHSV()

    def cnvt2HSV(self, verbose=False):
        """
        Converts BGR upperb and lowerb 2 HSV colors

        Arguments:
        ----------
        verbose (bool): prints details
        """
        return cv.cvtColor(np.uint8([[self.lowerb,self.upperb]]), cv.COLOR_BGR2HSV)

    def calcHSV(self,verbose=False):
        """
        Just calculates hsv of given data

        Arguments:
        ----------
        verbose (bool): prints details
        """
        return cv.cvtColor(self.data, cv.COLOR_BGR2HSV)

    ### TODO Make another detection via GRAYSCALE and Threshold values

    def calcBlur(self, verbose):
        """
        Converts data/frame/image to GaussianBlur-ed one.

        Arguments:
        ----------
        verbose (bool): prints details
        """
        if verbose:
            print("Calculating Blur")
        self.data = cv.GaussianBlur(self.data, (5,5), 0)
        self.hsv = self.calcHSV()

    def calcCnt(self, verbose=False):
        """
        Calculates contours

        Arguments:
        ----------
        verbose (bool): prints details
        """
        print("Calculating contours")
        mask = cv.inRange(self.hsv, self.hsv_colors[0][0], self.hsv_colors[0][1])
        kernel = np.ones((10,10),np.uint8)
        mask = cv.erode(mask, kernel)
        contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def drawContours(self, contours, showtxt=False,verbose=False):
        """
        Draws, contours and shows a text.

        Arguments:
        ----------
        showtxt (bool): shows text ıf set to true
        """
        x,y="",""
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .1*cv.arcLength(cnt,True), True)
            _area = cv.contourArea(approx)
            x,y = approx[0][0]
            x1= approx[1][0][0]
            appx1=int(float(x+x1)/2)
            if verbose:
                print("len: ",len(approx))
                print("contours: ",approx)
            if _area >= self.area and len(approx) == self.cornernum:
                M = cv.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                print("Center: {},{}".format(cx,cy))
                cv.circle(self.data, (cx,cy), 5, (0,0,0), 1)
                if verbose:
                    print("OK! Found @ "+str(x)+","+str(y))
                if showtxt:
                    self.data=cv.putText(self.data, "here", (appx1,y-20), cv.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 3, cv.LINE_AA)  
                cv.drawContours(self.data, [approx], -1, (0,0,0), 10)
        return (x,y)

    def detect(self,blur=False, showtxt=False,verbose=False):
        """
        Returns the shape boundaries drawen on image

        Arguments:
        ----------
        blur (bool): blurs the image for pure edges...
        showtxt (bool): shows text if set to True
        verbose (bool): shows every result... frame by frame...
        """
        if blur:
            self.calcBlur(verbose)
        contours=self.calcCnt(verbose)
        x,y=self.drawContours(contours,showtxt,verbose)
        return (self.data,x,y)
