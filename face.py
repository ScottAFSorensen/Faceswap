import numpy as np 
import cv2


def extract_face(convex_hull, image): 
    
    mask = np.zeros((image.shape[0], image.shape[1])) # gray level image
    
    cv2.fillConvexPoly(mask, convex_hull, 255)
    # print(type(mask))
    mask = mask.astype(np.uint8)
    face_region = cv2.bitwise_and(image, image, mask=mask)


    return mask, face_region
