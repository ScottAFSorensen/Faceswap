#! /usr/bin/env python3
import numpy as np 
import cv2
import math


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


def delaunay_triangulation(convex_hull, face1_landmarks, face2_landmarks, image, debug = False):
    '''
    :param convex_hull:     Convex hull of the first face
    :param face1_landmarks: Facial landmarks for the first face
    :param face2_landmarks: Facial landmarks for the second face
    :param image:           Image to draw the delaunay triangulation on
    :param debug:           Boolean to determine if debug is on, draws the delaunay triangles of the image if yes

    :return triangles1:     List with all the triangles in face1 as pixel coordinates
    :return triangles2:     List with all the triangles in face2 as pixel coordinates
    '''
    image_copy = None
    if debug:
        image_copy = np.copy(image)

    face1_landmarks_list = face1_landmarks.astype(int).tolist()
    face2_landmarks_list = face2_landmarks.astype(int).tolist()

    bounding_box = cv2.boundingRect(convex_hull)

    subdiv = cv2.Subdiv2D(bounding_box)
    subdiv.insert(face1_landmarks_list)

    # split face into triangles
    triang = subdiv.getTriangleList()
    triang = np.array(triang, dtype=np.int32)

    triangles_index = [] # the numbers seens in landmark_numbers.png
    triangles1 = [] # list of the triangles in face1
    triangles2 = [] # list of the triangles in face2

    for coordinate in triang:
        pt1 = (coordinate[0], coordinate[1])
        pt2 = (coordinate[2], coordinate[3])
        pt3 = (coordinate[4], coordinate[5])

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
        triangles_index.append(triangle_index) # remove later? do we need this list?

        # Corresponding triangle in second face
        pt1_2 = tuple(face2_landmarks_list[index1]) 
        pt2_2 = tuple(face2_landmarks_list[index2]) 
        pt3_2 = tuple(face2_landmarks_list[index3])
        
        # create list of the triangles
        face1_triangle = [pt1, pt2, pt3] 
        face2_triangle = [pt1_2, pt2_2, pt3_2]
        triangles1.append(face1_triangle) 
        triangles2.append(face2_triangle)

        if debug:
            cv2.line(image_copy, pt1, pt2, (0, 0, 255), 1)
            cv2.line(image_copy, pt2, pt3, (0, 255, 0), 1)
            cv2.line(image_copy, pt3, pt1, (255, 0, 0), 1)

            cv2.line(image_copy, pt1_2, pt2_2, (0, 0, 255), 1)
            cv2.line(image_copy, pt2_2, pt3_2, (0, 255, 0), 1)
            cv2.line(image_copy, pt3_2, pt1_2, (255, 0, 0), 1)


    if debug:
        cv2.imshow('Image with triangles', image_copy)
        cv2.waitKey()

    return [triangles1, triangles2]

def laplace_blend(image, swapped_image, mask1, mask2):
    '''
    :param image:           Original image from camera
    :param swapped_image:   Image with swapped faces
    :param mask1:           Mask for face1
    :param mask2:           Mask for face2

    :return blended_image:
    '''

    mask = mask1 + mask2 # image with both masks
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR) # turn mask to color image so shapes matches

    mask_f = 255 - mask # flipped mask.

    k = 10 # parameter for kernel size, have to find optimal value
    depth = 5 # depth of the pyramid

    gauss_mask = gaussian_pyramid(mask, depth, k) # gaussian pyramid of mask
    gauss_mask_f = gaussian_pyramid(mask_f, depth, k) # gaussian pyramid of flipped mask

    gauss_original = gaussian_pyramid(image, depth, k)
    gauss_swapped = gaussian_pyramid(swapped_image, depth, k)
    lap_pyr_original = laplace_pyramid(gauss_original, k) # Laplace pyramid of original image
    lap_pyr_swapped = laplace_pyramid(gauss_swapped, k) # Laplace pyramid of swapped faces image

    blended_pyr = []

    for i in range(len(gauss_mask)): 
        cv2.imshow('mask', gauss_mask[i])
        cv2.imshow('mask flipped', gauss_mask_f[i])
        cv2.imshow('lap_pyr_swapped[i]', lap_pyr_swapped[i])
        cv2.imshow('lap_pyr_original[i]', lap_pyr_original[i])

        cv2.waitKey()
        blended_pyr.append( gauss_mask[i]*lap_pyr_swapped[i] + gauss_mask_f[i]*lap_pyr_original[i] )# create the blended pyramid
    

    output = blended_pyr[len(blended_pyr)-1]
    for i in range(len(blended_pyr)-1, 0, -1):
        expanded = up2(output, k)

        output = expanded + blended_pyr[i - 1]

    cv2.imshow('blended', output)
    cv2.waitKey()


def down2(image, k):
    '''
    Used in laplace_blend()
    Blur and create image half the size.
    '''

    k_size = math.ceil((image.shape[0]/k)/ 2.)*2 + 1 # kernel size, make sure it's always a odd number
    gaussian_image = cv2.GaussianBlur(image ,(k_size,k_size),0) # gaussian of mask

    reduced = gaussian_image[::2, ::2] # downsample by 2

    return reduced

def up2(image, k):
    '''
    Used in laplace_blend()
    Create image twice as big
    '''

    upscaled = np.zeros((2*image.shape[0], 2*image.shape[1], 3)) # twice as big image
    upscaled[::2, ::2, :] = image

    upscaled = cv2.resize(image, (0,0), fx=2, fy=2) 

    cv2.imshow('up2', upscaled)
    cv2.waitKey()

    k_size = math.ceil((image.shape[0]/k)/ 2.)*2 + 1 
    gaussian_image = cv2.GaussianBlur(upscaled ,(k_size,k_size),0)

    upscaled = gaussian_image * 4 # multiplying the height and width by 2

    return upscaled


def gaussian_pyramid(image, depth, k):
    '''
    Used in laplace_blend()
    Using the down2 function iteratively to build a pyramid
    '''

    pyramid = [image]

    for i in range(depth):
        pyramid.append(down2(pyramid[i], k))

    return pyramid


def laplace_pyramid(gauss_pyr, k):
    '''
    Used in laplace_blend()
    Iterate the gaussian pyramid
    '''

    pyramid = []

    for i in range(len(gauss_pyr)-1):
        upscaled = up2(gauss_pyr[i+1], k)

        pyramid.append(gauss_pyr[i] - upscaled) # laplace image

    pyramid.append(gauss_pyr[len(gauss_pyr)-1])

    return pyramid
