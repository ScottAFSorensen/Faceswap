import cv2
import numpy as np
import dlib
 
#cap = cv2.VideoCapture(0) #0 built-in camera, 1 usb camera
cap = cv2.VideoCapture("Obama_and_Key.mp4")

#Using alread existing library for face detector and finding facial landmarks.
detector = dlib.get_frontal_face_detector() # face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Using pre-trained model to detect facial landmarks

FRAME = None # single frame taken from video

while(True):
    
    ret, frame = cap.read()        
    cv2.imshow('Video',frame)
 
    if cv2.waitKey(25) & 0xFF == ord('q'): #Q to quit
        break

    if cv2.waitKey(25) & 0xFF == ord(' '): #Spacebar, take image from video
        FRAME = frame
        break
        
cv2.destroyAllWindows() # close video

#----------------get facial landmarks part-----------------------
n_faces = 2
n_markers = 27
#markers 0-16 is jawline/face shape
#markers 17-26 eyebrows
#markers 27-67 nose, eyes, mouth
facial_landmarks = np.zeros((n_faces, n_markers, 2)) # create mat

gray = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY) #use detector on grayscale image
faces = detector(gray) #dlib

for m in range(0, n_faces):

    landmarks = predictor(gray, faces[m]) #dlib

    for n in range(0, n_markers):

        x = landmarks.part(n).x
        y = landmarks.part(n).y
        if m == 0:
            cv2.circle(FRAME, (x, y), 4, (0, 0, 255), -1) #visualize landmarks, BGR
        else: # just to vizually differentiate the faces
            cv2.circle(FRAME, (x, y), 4, (0, 255, 0), -1) 

        facial_landmarks[m, n, 0] = x
        facial_landmarks[m, n, 1] = y
        #cv2.putText(FRAME,str(n),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2,cv2.LINE_AA) # show marker number
    
cv2.imshow('Frame', FRAME)
cv2.waitKey()

#----------------------Convex hull part------------------------------------------

