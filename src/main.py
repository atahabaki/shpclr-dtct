import cv2 as cv
import numpy as np
from detect import Detective as dt

#"""
cap = cv.VideoCapture("../media/0000-0129.mkv")

while cap.isOpened():
    ret, frame = cap.read()
    wow = dt(np.array([130,130,130]), np.array([255,255,255]),4,90,frame)
    shit,x,y = wow.detect(showtxt=True, blur=True,verbose=True)
    if x != "" or y != "":
        print(x,y)
    cv.namedWindow("Frames", cv.WINDOW_NORMAL)
    cv.imshow("Frames", shit)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
#"""
"""
wow = dt(np.array([130,130,130]), np.array([255,255,255]),4,90,cv.imread("../media/out.jpg"))
frame,x,y = wow.get_x_y(True)
if x != "" or y != "":
    print(x,y)
"""
"""
wow = dt(np.array([100,100,100]), np.array([255,255,255]),4,90,cv.imread("../media/out2.png"))
frame,x,y = wow.get_x_y(True)
if x != "" or y != "":
    print(x,y)
"""
cv.destroyAllWindows()
