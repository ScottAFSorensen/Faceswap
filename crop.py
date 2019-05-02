import numpy as np 
import cv2


def get_face(convex_hull, image):
    # Find box region to crop
    box = cv2.boundingRect(convex_hull) 
    y,x,w,z = box

    cropped = image

    # Find mask (binary image)
    convex_hull = convex_hull - convex_hull.min(axis=0)

    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [convex_hull], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # bitwise and
    face_region = cv2.bitwise_and(cropped, cropped, mask=mask)

    return mask, face_region