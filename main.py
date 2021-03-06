#! /usr/bin/env python3
import cv2
import numpy as np
import dlib
from convex import get_hull
from face import extract_face, delaunay_triangulation, laplace_blend
from affine_trans import morph_affine

DEBUG = False # set True to run debug version of the program
cap = cv2.VideoCapture(0) #0 built-in camera, 1 usb camera

# Using already existing library for face detector and finding facial landmarks.
detector = dlib.get_frontal_face_detector()  # face detector
# Using pre-trained model to detect facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Make sure this is downloaded, see README

# ----------------facial landmarks-----------------------

n_faces = 2 # Code only works with 2 faces anyway at the moment
marker_start = 0
marker_end = 27
n_markers = marker_end - marker_start
# markers 0-16 is jawline/face shape
# markers 17-26 eyebrows
# markers 27-67 nose, eyes, mouth

# --------------- main face swap function -------------

def swap_faces(FRAME, gray, faces):
    '''
	:param FRAME:       Single image frame from the camera
	:param gray:        Grayscale image of FRAME

	:return swapp:      Face swapped image
	'''

    facial_landmarks = np.zeros((n_faces, n_markers, 2))  # create mat

    for m in range(0, n_faces):
        landmarks = predictor(gray, faces[m])  # dlib
        for n in range(marker_start, marker_end):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # cv2.circle(FRAME, (x, y), 4, (0, 0, 255), -1) #visualize landmarks, BGR

            facial_landmarks[m, n - marker_start] = (x, y)

    # ----------------------Convex hull (convex.py)------------------------------------------
    # Finds the convex hull of the faces, based on out own convex hull algorithm, (style of jarvis match)
    face1_hull = get_hull(facial_landmarks[0])
    face2_hull = get_hull(facial_landmarks[1])

    # ---------------------Extract face and mask (face.py)---------------------------------------
    face1_mask, face1 = extract_face(face1_hull, FRAME)
    face2_mask, face2 = extract_face(face2_hull, FRAME)

    if DEBUG:
        cv2.imshow('face1', face1)
        cv2.imshow('face2', face2)
        cv2.imshow('mask1', face1_mask)
        cv2.imshow('mask2', face2_mask)
    # --------------------- Delaunay triangulation (face.py)------------------------

    tri_face1_in_face2 = delaunay_triangulation(face1_hull, facial_landmarks[0], facial_landmarks[1], FRAME, DEBUG)
    tri_face2_in_face1 = delaunay_triangulation(face2_hull, facial_landmarks[1], facial_landmarks[0], FRAME, DEBUG)

    # --------------------Affine transform (affine_trans.py)----------------------------------------------

    swapp = np.copy(FRAME)
    if DEBUG:
        cv2.imshow('before affine transform and swapping', swapp)
        cv2.waitKey()

    for i in range(len(tri_face1_in_face2[0])):
        morph_affine(tri_face1_in_face2[0][i], tri_face1_in_face2[1][i], FRAME, swapp, DEBUG)

    for i in range(len(tri_face2_in_face1[0])):
        morph_affine(tri_face2_in_face1[0][i], tri_face2_in_face1[1][i], FRAME, swapp, DEBUG)

    if DEBUG:
        cv2.imshow('after affine transform and swapping', swapp)
        cv2.waitKey()

    # --------------------- Blur face edge, Laplace blending (face.py)----------------------------

    # Figure out blur amount:
    # Use facials landmarks 0 and 16 for width, see landmark_numbers.png
    width_face1 = abs(facial_landmarks[0][16][0] - facial_landmarks[0][0][0]) # width of face in pixels in horizontal direction.
    width_face2 = abs(facial_landmarks[1][16][0] - facial_landmarks[1][0][0])
    blur_size = int(((width_face1 + width_face2)/2)*0.5)
    if blur_size % 2 == 0:
        blur_size += 1

    swapp = laplace_blend(FRAME, swapp, face1_mask, face2_mask, blur_size)
    return swapp


# ----------------------- Video loop ----------------------
while True:
    swapped = None
    gray = None
    faces = None
    ret, FRAME = cap.read()
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Q to quit
        break

    gray = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)  # use detector on gray scale image, faster 
    faces = detector(gray)  # dlib


    if len(faces) >= 2:
        if DEBUG:
            if cv2.waitKey(25) & 0xFF == ord(' '):  # spacebar
                swapped = swap_faces(FRAME, gray, faces)
            else:
                cv2.imshow('Video', FRAME)
        else:
            swapped = swap_faces(FRAME, gray, faces)
            cv2.imshow('Video', swapped)
    else:
        cv2.putText(FRAME,"Can't find two faces", (0,FRAME.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Video', FRAME)
        
cv2.destroyAllWindows()  # close video


        
