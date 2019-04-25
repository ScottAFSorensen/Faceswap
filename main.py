import cv2
import numpy as np
import dlib
 
cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
 
while(cap.isOpened()):
    
    ret, frame = cap.read()

    #----------------Ferdig kode-----------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1) #visualize landmarks
    #--------------------------------------------------------------
        
    cv2.imshow('Video',frame)
 
    if cv2.waitKey(25) & 0xFF == ord('q'): #Q to quit
        break

    if cv2.waitKey(25) & 0xFF == ord(' '): #Spacebar, take image
        cv2.imshow('Image',frame)

        