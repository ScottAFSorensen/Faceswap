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


def delaunay_triangulation(convex_hull, face1_landmarks, face2_landmarks, image):
    '''
    :param convex_hull:     Convex hull of the first face
    :param face1_landmarks: Facial landmarks for the first face
    :param face2_landmarks: Facial landmarks for the second face
    :param image:           Image to draw the delaunay triangulation on, remove this param later maybe?
    '''

    face1_landmarks_list = face1_landmarks.astype(int).tolist()
    face2_landmarks_list = face2_landmarks.astype(int).tolist()

    bounding_box = cv2.boundingRect(convex_hull)

    subdiv = cv2.Subdiv2D(bounding_box)
    subdiv.insert(face1_landmarks_list)

    triang = subdiv.getTriangleList()
    triang = np.array(triang, dtype=np.int32)

    triangles_index = [] # the numbers seens in landmark_numbers.png
    triangles = []

    # For vizualisation
    for coordinate in triang:
        pt1 = (coordinate[0], coordinate[1])
        pt2 = (coordinate[2], coordinate[3])
        pt3 = (coordinate[4], coordinate[5])

        original_triangle = [pt1, pt2, pt3]

        # index chaos but it works :)
        index1 = np.where((face1_landmarks == pt1).all(axis=1))
        index1 = index1[0]
        if len(index1) == 0: # No triangle index
            continue
        index1 = index1[0]

        index2 = np.where((face1_landmarks == pt2).all(axis=1))
        index2 = index2[0]
        if len(index2) == 0: 
            continue
        index2 = index2[0] 

        index3 = np.where((face1_landmarks == pt3).all(axis=1))
        index3 = index3[0]
        if len(index3) == 0: 
            continue
        index3 = index3[0] 

        triangle_index = [index1, index2, index3]
        triangles_index.append(triangle_index)

        pt1_2 = tuple(face2_landmarks_list[index1]) # Corresponding triangle in second face
        pt2_2 = tuple(face2_landmarks_list[index2]) 
        pt3_2 = tuple(face2_landmarks_list[index3])

        
        triangle = [pt1_2, pt2_2, pt3_2]
        print("1", original_triangle)
        print("2", triangle)

        cv2.line(image, pt1, pt2, (0, 0, 255), 1)
        cv2.line(image, pt2, pt3, (0, 255, 0), 1)
        cv2.line(image, pt3, pt1, (255, 0, 0), 1)

        cv2.line(image, pt1_2, pt2_2, (0, 0, 255), 1)
        cv2.line(image, pt2_2, pt3_2, (0, 255, 0), 1)
        cv2.line(image, pt3_2, pt1_2, (255, 0, 0), 1)

    #print(triangles)
    return image, triangles_index, triangles

