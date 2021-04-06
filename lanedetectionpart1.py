import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('project_video.avi')

def undistortion(image):
    mtx=np.array([[9.037596e+02, 0, 6.957519e+02],[ 0, 9.019653e+02, 2.242509e+02],[ 0 , 0, 1]])
    dist=np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
    undistorted = cv2.undistort(frame,mtx,dist,None,mtx)
    return undistorted

def homography(image):
    pts1 = np.array([[477, 285], [740, 285], [260,405],[895,405]])
    pts2 = np.array([[0,0],[255,0],[0,255],[255,255]])
    H,status = cv2.findHomography(pts1, pts2)
    H_inv=np.linalg.inv(H)
    return  H, H_inv
def warpped(image):
    warpped = cv2.warpPerspective(image, H, (255,255))
    return warpped
def lanecolors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white  = np.array([ 0, 0, 175])
    upper_white = np.array([ 255, 80, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_line= cv2.bitwise_and(image,image,mask=mask_white)
    gray = cv2.cvtColor(white_line,cv2.COLOR_BGR2GRAY)
    #removing the noise
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #threhsolding the blurred image
    ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
    return thresh
def sobel1(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobel_final = np.sqrt((sobelx**2) + (sobely**2))
    return sobel_final
#step6 : finding the histogram of the bottom part of image frames
def hist(image):
    histo = np.sum(image[image.shape[0]//2:,:], axis=0)
    return histo
#step7 : implementing the sliding window algorithm
def sliding_window(histogram, image, num_of_windows=9, margin=50, minpix=10):
    frame1 = np.dstack((image,image,image)) * 255
    #fidning the midpoint
    center_point = np.int(histogram.shape[0]/2)
    #finding the peak values of left and right line bases of the histogram
    x_left_base = np.argmax(histogram[:center_point])
    x_right_base = np.argmax(histogram[center_point:]) + center_point
    #adjust the height of the window
    window_height = np.int(image.shape[0] / num_of_windows)
    #getting all the nonzero image pixels
    nonzero = image.nonzero()
    #seperating it into x and y
    y_nonzero = np.array(nonzero[0])
    x_nonzero = np.array(nonzero[1])
    #making the current window as the bottom of the histogram
    x_left_frame = x_left_base
    x_right_frame = x_right_base
    #lists to get pixel indices
    lane_pixels_left = []
    lane_pixels_right = []
    #sneaking through the windows
    for window in range(num_of_windows):
        #get the boundaries of the windows
        y_window_low = image.shape[0] - (window + 1) * window_height
        y_window_high = image.shape[0] - window * window_height

        x_left_low = x_left_frame - margin
        x_left_high = x_left_frame + margin

        x_right_low = x_right_frame - margin
        x_right_high = x_right_frame + margin

        cv2.rectangle(frame1,(x_left_low,y_window_low),(x_left_high,y_window_high),(0,255,0), 2)
        cv2.rectangle(frame1,(x_right_low,y_window_low),(x_right_high,y_window_high),(0,255,0), 2)

        #getting nonzero pixel values in the windows
        true_ind_left = ((y_nonzero >= y_window_low) & (y_nonzero < y_window_high) &
                        (x_nonzero >= x_left_low) & (x_nonzero < x_left_high)).nonzero()[0]
        true_ind_right = ((y_nonzero >= y_window_low) & (y_nonzero < y_window_high) &
                    (x_nonzero >= x_right_low) & (x_nonzero < x_right_high)).nonzero()[0]
        #append the indices
        lane_pixels_left.append(true_ind_left)
        lane_pixels_right.append(true_ind_right)
        #if the non zero values are greater than the predicted value
        if len(true_ind_left) > minpix:
            #recenter the next window as their mean position
            x_left_frame = np.int(np.mean(x_nonzero[true_ind_left]))
        if len(true_ind_right > minpix):
            x_right_frame = np.int(np.mean(x_nonzero[true_ind_right]))
    #Concatenating the left lane and right lane pixels
    lane_pixels_left = np.concatenate(lane_pixels_left)
    lane_pixels_right = np.concatenate(lane_pixels_right)
    x_left = x_nonzero[lane_pixels_left]
    y_left = y_nonzero[lane_pixels_left]
    x_right = x_nonzero[lane_pixels_right]
    y_right = y_nonzero[lane_pixels_right]
    return x_left, y_left, x_right, y_right, lane_pixels_left, lane_pixels_right
#step8 : polynomial fitting
def fitting_poly (x_left, y_left, x_right, y_right, lane_pixels_left, lane_pixels_right, sobel, a: True):
    #obtaining a new image
    frame1 = np.dstack((sobel,sobel, sobel))*255
    lane_left = np.polyfit(y_left, x_left, 2)
    lane_right= np.polyfit(y_right,x_right,2)
    #creating a sequence
    plt = np.linspace(0,sobel.shape[0]-1,sobel.shape[0])
    #getting x coordinatepixels
    lane_leftx = lane_left[0] * plt **2 + lane_left[1]*plt + lane_left[2]
    lane_rightx= lane_right[0] * plt**2 + lane_right[1]*plt +lane_right[2]
    #getting non zero pixel values from sobel
    nonzero = sobel.nonzero()
    y_nonzero = np.array(nonzero[0])
    x_nonzero = np.array(nonzero[1])

    #applying the polyfit
    frame1[y_nonzero[lane_pixels_left], x_nonzero[lane_pixels_left]] = [255, 255, 0]
    frame1[y_nonzero[lane_pixels_right], x_nonzero[lane_pixels_right]] = [0, 255, 255]
    #if true showing it in the frame
    if a:
        cv2.imshow("frame1",frame1)
    return frame1,lane_left, lane_right, plt, lane_leftx, lane_rightx

def plot_in_frame( image, sobel, Hinv, lane_leftx, lane_rightx, plt):
    poly = np.zeros_like(sobel).astype(np.uint8)
    #get the point format using y and y coordinate values
    left_values = np.array([np.transpose(np.vstack([lane_leftx, plt]))])
    right_values = np.array([np.flipud(np.transpose(np.vstack([lane_rightx, plt])))])
    values = np.hstack((left_values, right_values))
    #draw lane on the real time frames
    cv2.fillPoly(poly, np.int_([values]), (255,255,0))
    #warp the region in the real world frame
    warp_frame = cv2.warpPerspective(poly, Hinv, (image.shape[1], image.shape[0]))
    #putting up the text to predict the turns
    cv2.putText(warp_frame,"STRAIGHT",(400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)
    #add with the original frame
    output = cv2.addWeighted(image, 1, warp_frame, 0.3, 0)
    return output
while True:
    a=[]
    b=[]
    ret, frame =cap.read()
    undistorted_frame = undistortion(frame)
    H,H_inv = homography(undistorted_frame)
    warp = warpped(undistorted_frame)
    thresh = lanecolors(warp)
    sobel=sobel1(thresh)
    histogram = hist(sobel)
    x_left, y_left, x_right, y_right, lane_pixels_left, lane_pixels_right = sliding_window(histogram, sobel)
    if x_left!=a and x_right!=b:
        frame1,lane_left, lane_right, plt, lane_leftx, lane_rightx = fitting_poly(x_left, y_left, x_right, y_right, lane_pixels_left, lane_pixels_right,sobel,True)
        final=plot_in_frame(frame, frame1, H_inv, lane_leftx, lane_rightx, plt)
    cv2.imshow("final",final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
