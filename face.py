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
    facial_landmarks = facial_landmarks.tolist()

    bounding_box = cv2.boundingRect(convex_hull)
    print(bounding_box)
 
    subdiv = cv2.Subdiv2D(bounding_box)
    subdiv.insert(facial_landmarks)

    triang = subdiv.getTriangleList()
    triang = np.array(triang, dtype=np.int32)

    # For vizualisation
    for coordinate in triang:
        pt1 = (coordinate[0], coordinate[1])
        pt2 = (coordinate[2], coordinate[3])
        pt3 = (coordinate[4], coordinate[5])

        cv2.line(image, pt1, pt2, (0, 0, 255), 1)
        cv2.line(image, pt2, pt3, (0, 255, 0), 1)
        cv2.line(image, pt3, pt1, (255, 0, 0), 1)

        #index1 = np.where((facial_landmarks == pt1).all(axis=1))
        #index1 = extract_index_nparray(index1)

        #index2 = np.where((facial_landmarks == pt2).all(axis=1))
        #index2 = extract_index_nparray(index2)

        #index3 = np.where((facial_landmarks == pt3).all(axis=1))
        #index3 = extract_index_nparray(index3)
    
    return image

def extract_index_nparray(nparray):
    index = None

    for num in nparray[0]:

        index = num
        break
    return index
