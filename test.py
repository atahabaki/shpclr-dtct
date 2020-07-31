import cv2 as cv
import numpy as np

class Shape:
    def __init__(self, tmin, tmax, amin, amax, lowerb=None, upperb=None, cmin=0,cmax=None):
        """
        Args:
        =====
        tmin: Threshold min val
        tmax: Threshold min val
        amin: min val of the object's area
        amax: max val of the object's area
        lowerb: BGR color code of the object's color
        upperb: BGR color code of the object's color
        cmin: minimum number of corners the object has
        cmax: maximum number of corners the object has
        """
        self.tmin = tmin
        self.tmax = tmax
        self.amin = amin
        self.amax = amax
        self.lowerb = lowerb
        self.upperb = upperb
        self.cmin = cmin
        self.cmax = cmax

class Detective:
    def __init__(self, shape, data, erodeSize=(5,5), verbose=True):
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
        self.erodeSize=erodeSize

    def resizeImg(self, dimens):
        return cv.resize(self.data,dimens)
    def blurImg(self, img,i=1):
        for x in range(i):
            img = cv.GaussianBlur(img, (7,7), 1)
        return img
    def detectByThresholding(self,dimens=(720,405),drawContours=True,inverted=False):
        """
        Detects object returns resized and blurred image, and X,Y coordinates of detected object
        Returns:
        ========
        (image, x, y)
        Arguments:
        ==========
        dimens (width, height): Determines the resize options just width & height
        inverted (bool): if set to True Threshold would be calculated by THRESH_BINARY_INV
        drawContours (bool): Draws contours if set to True
        """
        img = self.resizeImg(dimens)
        img = self.blurImg(img)
        gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        method = cv.THRESH_BINARY
        if inverted:
            method = cv.THRESH_BINARY_INV
        _, thresh = cv.threshold(gry, self.shape.tmin, self.shape.tmax, method)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cx,cy=None,None
        if len(contours) > 0:
            for i,cnt in enumerate(contours,start=0):
                approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
                area = cv.contourArea(approx)
                if self.verbose:
                    print("contours:",len(approx))
                    print("area",area)
                if area >= self.shape.amin and area <= self.shape.amax:
                    M = cv.moments(approx)
                    if M['m00'] != 0:
                        cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                    print("center:",(cx,cy))
                    if drawContours:
                        cv.drawContours(img,[approx],i,(0,0,0),10)
        return (img, cx, cy, thresh, gry)
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
            mask = cv.erode(mask, np.ones(self.erodeSize,np.uint8))
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cx,cy=None,None
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
            area = cv.contourArea(approx)
            if self.verbose:
                print("contours:",len(approx))
                print("area:",area)
            if area >= self.shape.amin and area <= self.shape.amax:
                M = cv.moments(approx)
                if M['m00'] != 0:
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
            mask = cv.erode(mask, np.ones(self.erodeSize,np.uint8))
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cx,cy=None,None
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
            area = cv.contourArea(approx)
            if self.verbose:
                print("contours:",len(approx))
                print("area:",area)
            M = cv.moments(approx)
            if M['m00'] != 0:
                cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
            print("center:",(cx,cy))
            if drawContours:
                cv.drawContours(img,[approx],-1,(0,0,0),10)
        return (img, cx, cy)
    def detectByHoughCircles(self,dimens=(720,405),erode=True,drawContours=True):
        """
        !Not Recommended
        !Experimental
        Detects image by Canny Edge Detection Algorithm
        Arguments:
        ==========
        dimens (width, height): Determines the resize options just width & height
        erode (bool): Removes noises smaller or equal to 20x20 if set to True
        drawContours (bool): Draws contours if set to True
        apertureSize (int 3-7): apertureSize of Canny edge detection algorithm...
        """
        #img = self.resizeImg(dimens)
        img = self.data
        img = self.blurImg(img,5)
        gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if erode:
            img = cv.erode(img, np.ones(self.erodeSize,np.uint8))
        circles = cv.HoughCircles(gry, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0, maxRadius=0)
        cx,cy=None,None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cx,cy=center
                cv.circle(img, center, 1, (0, 100, 100), 3)
                radius = i[2]
                cv.circle(img, center, radius, (255, 0, 255), 3)
        if self.verbose:
            cv.imshow("circles",img)
        return (img,cx,cy)
    def detectByCanny(self,dimens=(720,405),erode=True,drawContours=True,apertureSize=4,inverted=True,numWhite=50):
        """
        !Not Recommended
        !Experimental
        Detects image by Canny Edge Detection Algorithm
        Arguments:
        ==========
        dimens (width, height): Determines the resize options just width & height
        erode (bool): Removes noises smaller or equal to 20x20 if set to True
        drawContours (bool): Draws contours if set to True
        apertureSize (int 3-7): apertureSize of Canny edge detection algorithm...
        Experimental:
        =============
        numWhite (int): maximum difference between masks...
        """
        img = self.resizeImg(dimens)
        img = self.blurImg(img)
        gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if erode:
            img = cv.erode(img, np.ones(self.erodeSize,np.uint8))
        edges = cv.Canny(gry, self.shape.tmin, self.shape.tmax, apertureSize, L2gradient=True)
        cx,cy=None,None
        if self.verbose:
            cv.imshow("edges",edges)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
            area = cv.contourArea(approx)
            if area <= self.shape.amax and area >= self.shape.amin and len(approx) <= self.shape.cmax and len(approx) >= self.shape.cmin:
                if self.verbose:
                    print("contours:",len(approx))
                    print("area:",area)
                M = cv.moments(approx)
                if M['m00'] != 0:
                    cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                print("center:",(cx,cy))
                if drawContours:
                    cv.drawContours(img,[approx],-1,(0,0,0),10)
        return (img,cx,cy)
    def detectByComplexAlgorithm(self,dimens=(720,405),erode=True,drawContours=True,inverted=True,numWhite=50):
        """
        !Experimental
        Detects by number of corners (basically shape), color, area, threshold...
        Also, compares color mask and threshold mask for better accuracy...
        May result performance issues...
        Arguments:
        ==========
        dimens (width, height): Determines the resize options just width & height
        erode (bool): Removes noises smaller or equal to 20x20 if set to True
        drawContours (bool): Draws contours if set to True
        inverted (bool): if set to True Threshold would be calculated by THRESH_BINARY_INV
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
        method = cv.THRESH_BINARY
        if inverted:
            method = cv.THRESH_BINARY_INV
        _, threshold = cv.threshold(gry, self.shape.tmin, self.shape.tmax, method)
        cv.imshow("mask0",mask0)
        cv.imshow("threshold",threshold)
        subtracted = cv.subtract(mask0,threshold)
        cv.imshow("subtracted",subtracted)
        #print("Array Len:",len(subtracted))
        whites = np.count_nonzero(subtracted == 255)
        if self.verbose:
            print("whites:",whites)
        cx,cy=None,None
        if whites <= numWhite:
            contours, _ = cv.findContours(mask0, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                approx = cv.approxPolyDP(cnt, .001*cv.arcLength(cnt, True), True)
                area = cv.contourArea(approx)
                if area <= self.shape.amax and area >= self.shape.amin and len(approx) <= self.shape.cmax and len(approx) >= self.shape.cmin:
                    if self.verbose:
                        print("contours:",len(approx))
                        print("area:",area)
                    M = cv.moments(approx)
                    if M['m00'] != 0:
                        cx, cy = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                    print("center:",(cx,cy))
                    if drawContours:
                        cv.drawContours(img,[approx],-1,(0,0,0),10)
        return (img, cx, cy)

def whoo(x):
    pass

#FOR GRAY shitty box:
#LRGB: 55,62,75
#URGB: 143,143,143
#MinArea: 130000
#MaxArea: 150000
#LTHRESH: 0
#UTHRESH: 255
#cmin: 0
#cmax: 100

#FOR RED BALL
# MinArea: 500
# MaxArea: 200000
#LTHRESH: 0
#UTHRESH: 255
#cmin: 80
#cmax: 130

cv.namedWindow("TBARS", cv.WINDOW_NORMAL)
cv.createTrackbar("LThresh", "TBARS", 0, 255, whoo)
cv.createTrackbar("UThresh", "TBARS", 255, 255, whoo)
cv.createTrackbar("minArea", "TBARS", 130000, 200000, whoo)
cv.createTrackbar("maxArea", "TBARS", 150000, 200000, whoo)
cv.createTrackbar("cmin", "TBARS", 0, 100, whoo)
cv.createTrackbar("cmax", "TBARS", 200, 200, whoo)
cv.createTrackbar("numWhite", "TBARS", 140000, 200000, whoo)
cv.namedWindow("RGB", cv.WINDOW_NORMAL)
cv.createTrackbar("LR", "RGB", 55, 255, whoo)
cv.createTrackbar("LB", "RGB", 62, 255, whoo)
cv.createTrackbar("LG", "RGB", 75, 255, whoo)
cv.createTrackbar("UR", "RGB", 143, 255, whoo)
cv.createTrackbar("UB", "RGB", 143, 255, whoo)
cv.createTrackbar("UG", "RGB", 143, 255, whoo)

cap = cv.VideoCapture(2)

while True:
    lb = cv.getTrackbarPos("LThresh", "TBARS")
    ub = cv.getTrackbarPos("UThresh", "TBARS")
    mina = cv.getTrackbarPos("minArea", "TBARS")
    maxa = cv.getTrackbarPos("maxArea", "TBARS")
    cmin = cv.getTrackbarPos("cmin", "TBARS")
    cmax = cv.getTrackbarPos("cmax", "TBARS")
    numw = cv.getTrackbarPos("numWhite", "TBARS")
    lb = cv.getTrackbarPos("LB","RGB")
    lg = cv.getTrackbarPos("LG","RGB")
    lr = cv.getTrackbarPos("LR","RGB")
    ub = cv.getTrackbarPos("UB","RGB")
    ug = cv.getTrackbarPos("UG","RGB")
    ur = cv.getTrackbarPos("UR","RGB")
    shape = Shape(lb,ub,mina,maxa,np.array([lb,lg,lr]),np.array([ub,ug,ur]),cmin,cmax)
    ret, fra = cap.read()
    #fra = cv.cvtColor(fra, cv.COLOR_BGR2GRAY)
    detective = Detective(shape,fra)
    #fra, cx, cy = detective.detectByComplexAlgorithm(numWhite=numw,inverted=True)
    #fra, cx, cy = detective.detectByColorOnly()
    #fra, cx, cy = detective.detectByCanny(dimens=(400,300),apertureSize=7)
    #fra = cv.resize(fra,(640,480))
    #fra = cv.GaussianBlur(fra, (7,7), 1)
    #detective.detectByHoughCircles()
    #mask = cv.inRange(fra,np.array([90,140,80]),np.array([]))
    cv.imshow("fra",fra)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
"""
shape = Shape(100,255,60000,100000,np.array([100,100,100]),np.array([180,180,180]))
detective = Detective(shape,cv.imread("media/out2.png"))
#img, cx, cy = detective.detectByColorOnly()
detective.detectByComplexAlgorithm()

#cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
"""
"""
shape = Shape(150,255,100,100000)
detective = Detective(shape,cv.imread("media/out3.png"))
img, cx, cy, thresh, gry = detective.detectByThresholding()

cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
"""
