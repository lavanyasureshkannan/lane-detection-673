import numpy as np
import cv2
cap = cv2.VideoCapture("video673.mp4")

def histogram(frame):
    h=np.zeros(shape=(256,1))
    s= frame.shape
    for i in range(s[0]):
        for j in range(s[1]):
            a = frame[i,j]
            h[a,0] = h[a,0] + 1
    return h

while True:
    ret,frame= cap.read()
    frame= cv2.resize(frame,(500,500))
    s=frame.shape
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray= cv2.convertScaleAbs(frame, alpha=4, beta=30)
    hist=histogram(gray)

    x=hist.reshape(1,256)
    y=np.array([])
    y=np.append(y,x[0,0])

    for i in range(255):
        m=x[0,i+1]+y[i]
        y=np.append(y,m)
    y=np.round((y/s[0] * s [1])) * (256-1)

    for i in range(s[0]):
        for j in range(s[1]):
            m=gray[i,j]
            gray[i,j]=y[m]
    cv2.imshow('gray',gray)
    #cv2.imshow('hist',histg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
