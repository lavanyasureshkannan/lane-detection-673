import numpy as np
import cv2
import math
cap = cv2.VideoCapture('challenge_video.mp4')
print(frame.shape)
#writer= cv2.VideoWriter('challenge.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (1,height))
#step1 : undistorting the frames
def undistortion(image):
    mtx=np.array([[1154.227,  0 , 671.628],[  0, 1148.182 , 386.0463],[ 0 , 0, 1]])
    dist=np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
    undistorted = cv2.undistort(frame,mtx,dist,None,mtx)
    return undistorted
#step2 : finding the homography matrix and inverse homography
def homography(image):
    pts1 = np.array([[550,450], [730, 450], [100,710],[1265, 710]])
    pts2 = np.array([[0,0],[255,0],[0,255],[255,255]])
    H,status = cv2.findHomography(pts1, pts2)
    H_inv=np.linalg.inv(H)
    return  H, H_inv
#step3: warpping the frames to get parallel lanes
def warpped(image):
    warpped = cv2.warpPerspective(image, H, (255,255))
    return warpped
#step4: getting the lane colors
def lanecolors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #threhsolding the yellow color line
    lower_yellow  = np.array([ 0, 85, 110])
    upper_yellow = np.array([ 40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_line = cv2.bitwise_and(image,image,mask=mask_yellow)
    #thresholding the white color line
    lower_white  = np.array([ 0, 0, 175])
    upper_white = np.array([ 255, 80, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_line= cv2.bitwise_and(image,image,mask=mask_white)
    #adding both
    final_line = cv2.bitwise_or(yellow_line,white_line)
    #converting to gray scale
    gray = cv2.cvtColor(final_line,cv2.COLOR_BGR2GRAY)
    #removing the noise
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #threhsolding the blurred image
    ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
    return thresh
#step5:using sobel function to extract the edges
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
#step9 : predicting the turns
def predict_trun(lane_left, lane_right, plt, lane_leftx, lane_rightx, x_left, y_left, x_right, y_right, sobel):
        ev_y = np.max(plt)
        #assumin camera center
        cam_mid = sobel.shape[1] / 2
        #meters per pixel in y dimension
        ym_per_pixel_ = 30/720
        #meters per pixel in x dimension assuming the road length as 3.7 meters
        xm_per_pixel = 3.7/720
        #using polyfit find x coordinate left lane values
        left_center = np.polyfit(plt * ym_per_pixel_, lane_leftx * xm_per_pixel, 2)
        left_curve= ((1 + (2 * lane_left[0] * ev_y / 2 + left_center[1]) ** 2) ** 1.5) / np.absolute(2 * left_center[0])
        #using polyfit find y coordinate right lane values
        right_center = np.polyfit(plt * ym_per_pixel_, lane_rightx * xm_per_pixel,2)
        right_curve = ((1 + (2 * lane_left[0] * ev_y / 2 + right_center[1]) ** 2) ** 1.5) / np.absolute(2 * right_center[0])
        #getting the middle of left and right lane bottoms
        lane_mid = (lane_leftx[0] + lane_rightx[0]) / 2
        #getting middle pixel values
        pixel_Center = lane_mid - cam_mid
        #convert into meters
        center_meter = pixel_Center * xm_per_pixel
        #getting the curvature radius in meter.
        turn = ""
        if center_meter <= -0.1:
            turn = "left"
        elif center_meter >= 0.1:
            turn = "right"
        else:
            turn = "straight"
        return turn
#step10: bringing up all the values in the real world
def plot_in_frame(turn, image, sobel, Hinv, lane_leftx, lane_rightx, plt):
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
    cv2.putText(warp_frame,str(turn),(600, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2,cv2.LINE_AA)
    #add with the original frame
    output = cv2.addWeighted(image, 1, warp_frame, 0.3, 0)
    return output
#while the frames are read apply all the functions describes above step by step
while True:
    a=[]
    b=[]
    _,frame = cap.read()
    undistorted_frame = undistortion(frame)
    H,H_inv = homography(frame)
    warp = warpped(frame)
    thresh = lanecolors(warp)
    sobel=sobel1(thresh)
    histogram = hist(sobel)
    x_left, y_left, x_right, y_right, lane_pixels_left, lane_pixels_right = sliding_window(histogram, sobel)
    if x_left!=a and x_right!=b:
        frame1,lane_left, lane_right, plt, lane_leftx, lane_rightx = fitting_poly(x_left, y_left, x_right, y_right, lane_pixels_left, lane_pixels_right,sobel,True)
        predicted_turn = predict_trun(lane_left, lane_right, plt, lane_leftx, lane_rightx, x_left, y_left, x_right, y_right,sobel)
        final=plot_in_frame(predicted_turn,frame, frame1, H_inv, lane_leftx, lane_rightx, plt)
        cv2.imshow("final",final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
