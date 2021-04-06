import numpy as np
import cv2
cap = cv2.VideoCapture("video673.mp4")
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter('imrpoved.avi', fourcc, 30.0, (1000,600))
#writer= cv2.VideoWriter('improved_quality.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (800,800))
def gamma_correction(image, gamma = 1.0):
    gamma_inv = 1.0 / gamma
    dict = np.array([((i / 255.0) ** gamma_inv) * 255
        for i in np.arange(0,256)])
    return cv2.LUT(image.astype(np.uint8),dict.astype(np.uint8))
while True:
    ret,frame = cap.read()
    frame1 = cv2.resize(frame,(900,700))
    blur = cv2.GaussianBlur(frame1,(5,5),0)
    corrected= gamma_correction(blur, gamma=1.4)
    gray = cv2.convertScaleAbs(corrected, alpha=3, beta=15)
    #cv2.imshow("corrected",corrected)
    #cv2.imshow("frame1",frame1)

    cv2.imshow("gray",gray)
    videoWriter.write(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#writer.release()
cap.release()
cv2.destroyAllWindows()
