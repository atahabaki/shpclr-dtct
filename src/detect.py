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
    def __init__(self, shape, data, verbose=True):
        """
        Args:
        =====
        shape (Shape): determines the specifications of the object which will be searched
        data (Img): Mat object in C, which is basically an image data...
        """
        if type(shape) == Shape:
            self.shape = shape
        else:
            raise "unk. type shape"
        self.data = data
        self.verbose = verbose

    def detectViaThreshold(self,picSize=(720,405),drawContours=True):
        """
        Detects object returns resized and blurred image, and X,Y coordinates of detected object
        Returns:
        ========
        (image, x, y)
        Arguments:
        ==========
        picSize (width, height): Determines the resize options just width & height
        drawContours (bool): Draws contours if set to True
        """
        img = cv.resize(self.data,picSize)
        img = cv.GaussianBlur(img,(7,7),1)
        gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gry, self.shape.tmin, self.shape.tmax, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cx,cy=None,None
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
            area = cv.contourArea(approx)
            if self.verbose:
                print("area",area)
            if area <= self.shape.amax and area >= self.shape.amin:
                M = cv.moments(approx)
                cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                print("center:",(cx,cy))
                if drawContours:
                    cv.drawContours(img,[approx],-1,(0,0,0),10)
        return (img, cx, cy)
