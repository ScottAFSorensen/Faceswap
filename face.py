import numpy as np 
import cv2


def extract_face(convex_hull, image): 
    
    mask = np.zeros((image.shape[0], image.shape[1])) # gray level image
    
    cv2.fillConvexPoly(mask, convex_hull, 1)
    # print(type(mask))
    mask = mask.astype(np.uint8)

    face_region = np.zeros_like(image)
    face_region[:,:,0] = image[:,:,0]*mask # B
    face_region[:,:,1] = image[:,:,1]*mask # G
    face_region[:,:,2] = image[:,:,2]*mask # R

    mask = mask*255 # show mask as white

    return mask, face_region


def delaunay_triangulation(convex_hull, facial_landmarks, image):

    # WIP
    bounding_box = cv2.boundingRect(convex_hull)
    subdiv = cv2.Subdiv2D(bounding_box)
    subdiv.insert(facial_landmarks)

    triang = subdiv.getTriangleList()
    triang = np.array(triang, dtype=np.int32)

    for t in triang:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)
        cv2.line(image, pt2, pt3, (0, 0, 255), 2)
        cv2.line(image, pt1, pt3, (0, 0, 255), 2)
    
    return image
