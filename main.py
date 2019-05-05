import cv2
import numpy as np
import dlib
from convex import get_hull
from face import extract_face, delaunay_triangulation
 
# cap = cv2.VideoCapture(0) #0 built-in camera, 1 usb camera
cap = cv2.VideoCapture("Obama_and_Key.mp4") 

# Using already existing library for face detector and finding facial landmarks.
detector = dlib.get_frontal_face_detector() # face detector
# Using pre-trained model to detect facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

FRAME = None # single frame taken from video

while(True):
    
    ret, frame = cap.read()        
    cv2.imshow('Video',frame)
 
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Q to quit
        break

    if cv2.waitKey(25) & 0xFF == ord(' '):  # Spacebar, take image from video
        FRAME = frame
        break
        
cv2.destroyAllWindows() # close video

# ----------------get facial landmarks-----------------------
# Should be the length of predictor, can change it later
n_faces = 2
marker_start = 0
marker_end = 27
n_markers = marker_end - marker_start
# markers 0-16 is jawline/face shape
# markers 17-26 eyebrows
# markers 27-67 nose, eyes, mouth

facial_landmarks = np.zeros((n_faces, n_markers, 2)) # create mat

gray = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)  # use detector on gray scale image
faces = detector(gray)  # dlib

for m in range(0, n_faces):

    landmarks = predictor(gray, faces[m])  # dlib

    for n in range(marker_start, marker_end):

        x = landmarks.part(n).x
        y = landmarks.part(n).y
        # cv2.circle(FRAME, (x, y), 4, (0, 0, 255), -1) #visualize landmarks, BGR

        facial_landmarks[m, n-marker_start] = (x,y)

        # cv2.putText(FRAME,str(n),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2,cv2.LINE_AA) # show marker number
    
#cv2.imshow('Frame', FRAME)
#cv2.waitKey()

# ----------------------Convex hull (convex.py)------------------------------------------

# Finds the convex hull of the faces, based on out own convex hull algorithm, (style of jarvis match)
face1_hull = get_hull(facial_landmarks[0])
face2_hull = get_hull(facial_landmarks[1])


# ---------------------Extract face and mask (crop.py)---------------------------------------

face1_mask, face1 = extract_face(face1_hull, FRAME)
face2_mask, face2 = extract_face(face2_hull, FRAME)
cv2.imshow('face1', face1)
cv2.imshow('face2', face2)
#cv2.imshow('mask1', face1_mask)
#cv2.imshow('mask2', face2_mask)
cv2.waitKey()


# --------------------- Trying to find the delauney triangulation, using packages ------------------------

triang_image1 = delaunay_triangulation(face1_hull, facial_landmarks[0], FRAME)
cv2.imshow('delaunay', triang_image1)
cv2.waitKey()




# --------------------Affine transform----------------------------------------------




