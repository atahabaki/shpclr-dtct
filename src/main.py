import cv2 as cv
import numpy as np
from detect import Detective as dt

#"""
cap = cv.VideoCapture("../media/0000-0129.mkv")

while cap.isOpened():
    ret, frame = cap.read()
    wow = dt(np.array([130,130,130]), np.array([255,255,255]),4,frame)
    shit,x,y = wow.get_x_y()
    if x != "" or y != "":
        print(x,y)
    cv.namedWindow("Frames", cv.WINDOW_NORMAL)
    cv.imshow("Frames", shit)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
#"""
"""
wow = dt(np.array([130,130,130]), np.array([255,255,255]),4,cv.imread("../media/out.jpg"))
wow.get_x_y(True)
"""
"""
wow = dt(np.array([100,100,100]), np.array([255,255,255]),4,cv.imread("../media/out2.png"))
wow.get_x_y(True)
"""
cv.destroyAllWindows()
