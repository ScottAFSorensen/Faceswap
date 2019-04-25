import cv2
import numpy as np
import dlib
 
cap = cv2.VideoCapture(1)
 
while(cap.isOpened()):
    
    ret, frame = cap.read()

    cv2.imshow('Video',frame)
 
    if cv2.waitKey(25) & 0xFF == ord('q'): #Q to quit
        break

    if cv2.waitKey(25) & 0xFF == ord(' '): #Spacebar, take image

        cv2.imshow('Image',frame)

        
 

 
#Release and exit
cap.release()
cv2.destroyAllWindows()