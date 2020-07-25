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

    def resizeImg(self, dimens):
        return cv.resize(self.data,dimens)
    def blurImg(self, img):
        return cv.GaussianBlur(img,(7,7),1)
    def detectByThresholding(self,dimens=(720,405),drawContours=True):
        """
        Detects object returns resized and blurred image, and X,Y coordinates of detected object
        Returns:
        ========
        (image, x, y)
        Arguments:
        ==========
        dimens (width, height): Determines the resize options just width & height
        drawContours (bool): Draws contours if set to True
        """
        img = self.resizeImg(dimens)
        img = self.blurImg(img)
        gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gry, self.shape.tmin, self.shape.tmax, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cx,cy=None,None
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
            area = cv.contourArea(approx)
            if self.verbose:
                print("area",area)
            if area >= self.shape.amin and area <= self.shape.amax:
                M = cv.moments(approx)
                cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                print("center:",(cx,cy))
                if drawContours:
                    cv.drawContours(img,[approx],-1,(0,0,0),10)
        return (img, cx, cy)
    def detectByColor(self,dimens=(720,405),erode=True,drawContours=True):
        """
        Detects the object by it's color and approximate area
        Returns:
        ========
        (image, x, y)
        Arguments:
        ==========
        dimens (width, height): Determines the resize options just width & height
        erode (bool): Removes noises smaller or equal to 20x20 if set to True
        drawContours (bool): Draws contours if set to True
        """
        img = self.resizeImg(dimens)
        img = self.blurImg(img)
        mask = cv.inRange(img, self.shape.lowerb, self.shape.upperb)
        if erode:
            mask = cv.erode(mask, np.ones((20,20),np.uint8))
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cx,cy=None,None
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
            area = cv.contourArea(approx)
            if self.verbose:
                print("area",area)
            if area >= self.shape.amin and area <= self.shape.amax:
                M = cv.moments(approx)
                cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                print("center:",(cx,cy))
                if drawContours:
                    cv.drawContours(img,[approx],-1,(0,0,0),10)
        return (img, cx, cy)
    def detectByColorOnly(self,dimens=(720,405),erode=True,drawContours=True):
        """
        Detects the object by it's color only
        Returns:
        ========
        (image, x, y)
        Arguments:
        ==========
        dimens (width, height): Determines the resize options just width & height
        erode (bool): Removes noises smaller or equal to 20x20 if set to True
        drawContours (bool): Draws contours if set to True
        """
        img = self.resizeImg(dimens)
        img = self.blurImg(img)
        mask = cv.inRange(img, self.shape.lowerb, self.shape.upperb)
        if erode:
            mask = cv.erode(mask, np.ones((20,20),np.uint8))
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cx,cy=None,None
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
            area = cv.contourArea(approx)
            if self.verbose:
                print("area",area)
            M = cv.moments(approx)
            cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
            print("center:",(cx,cy))
            if drawContours:
                cv.drawContours(img,[approx],-1,(0,0,0),10)
        return (img, cx, cy)

    def detectByComplexAlgorithm(self,dimens=(720,405),erode=True,drawContours=True,numWhite=50):
        """
        !!!Not tested yet.
        Detects by number of corners (basically shape), color, area, threshold...
        Also, compares color mask and threshold mask for better accuracy...
        May result performance issues...
        Arguments:
        ==========
        dimens (width, height): Determines the resize options just width & height
        erode (bool): Removes noises smaller or equal to 20x20 if set to True
        drawContours (bool): Draws contours if set to True
        Experimental:
        =============
        numWhite (int): maximum difference between masks...
        """
        img = self.resizeImg(dimens)
        img = self.blurImg(img)
        if erode:
            img = cv.erode(img, np.ones(self.erodeSize,np.uint8))
        gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask0 = cv.inRange(img, self.shape.lowerb, self.shape.upperb)
        _, threshold = cv.threshold(gry, self.shape.tmin, self.shape.tmax, cv.THRESH_BINARY)
        cv.imshow("mask0",mask0)
        cv.imshow("threshold",threshold)
        subtracted = cv.subtract(mask0,threshold)
        print("Array Len:",len(subtracted))
        whites = np.count_nonzero(subtracted == 255)
        if whites <= numWhite:
            contours, _ = cv.findContours(mask0, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cx,cy=None,None
            for i,cnt in enumerate(contours,start=0):
                approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
                area = cv.contourArea(approx)
                if self.verbose:
                    print("contours:",len(approx))
                    print("area:",area)
                M = cv.moments(approx)
                cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                print("center:",(cx,cy))
                if drawContours:
                    cv.drawContours(img,[approx],i,(0,0,0),10)
        return (img, cx, cy)
