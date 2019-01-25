#testing the lucas kanade function on a video
import numpy as np 
import cv2
import matplotlib
from LK import lucas_kanade, compute_flow_map


cap = cv2.VideoCapture("video/walking.avi")
firstFrame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("flow map", cv2.WINDOW_NORMAL)
current_frame = firstFrame
previous_frame = current_frame

while True:

    ret, frame = cap.read()

    if ret == True:

        
        cv2.imshow('frame',frame)

        #finding the optical flow between two consecutive frames 
        u, v = lucas_kanade(current_frame, previous_frame)
        flow_map = compute_flow_map(u, v)
        cv2.imshow('flow map', flow_map.astype(firstFrame.dtype))


        #update the previous and current frames
        previous_frame = current_frame  
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
