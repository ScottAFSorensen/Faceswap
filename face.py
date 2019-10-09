#! /usr/bin/env python3
import numpy as np 
import cv2
import math

def extract_face(convex_hull, image): 
	'''
	:param convex_hull:     Convex hull of the face
	:param image:           Image of face

	:return mask:			Mask of face region
	:return face_region:    Image of just face region
	'''
	
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
	:param debug:           Set True to run in "debug" mode

	:return triangles1:     List with all the triangles in face1 as pixel coordinates
	:return triangles2:     List with all the triangles in face2 as pixel coordinates
	'''
	image_copy = None
	if debug:
		image_copy = np.copy(image)

	face1_landmarks_list = face1_landmarks.astype(int).tolist()
	face2_landmarks_list = face2_landmarks.astype(int).tolist()

	bounding_box = cv2.boundingRect(convex_hull) # Area to draw trangles in

	subdiv = cv2.Subdiv2D(bounding_box)
	subdiv.insert(face1_landmarks_list) # insert the points

	# split face into triangles
	triang = subdiv.getTriangleList()
	triang = np.array(triang, dtype=np.int32)

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

def laplace_blend(image, swapped_image, mask1, mask2, k_size):
	'''
	:param image:           Original image from camera
	:param swapped_image:   Image with swapped faces
	:param mask1:           Mask for face1
	:param mask2:           Mask for face2
	:param k_size:          Kernel size for gaussian blur

	:return blended_image:
	'''

	mask = mask1 + mask2 # image with both masks
	mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR) # turn mask to color image so shapes matches
	mask = cv2.GaussianBlur(mask ,(k_size,k_size),0) # gaussian of mask
	
	# needed to divide by 255 to normalize the values because cv2 is stupid
	image = image.astype(float) / 255 
	swapped_image = swapped_image.astype(float) / 255 
	mask = mask.astype(float) / 255

	mask_f = 1 - mask # flipped mask.
	
	# linear_blend = mask*swapped_image + mask_f*image # simple linear blending
   
	depth = 5 # depth of the pyramids
	gauss_pyr_mask = gaussian_pyramid(mask, depth) # gaussian pyramid of mask
	gauss_pyr_mask_f = gaussian_pyramid(mask_f, depth) # gaussian pyramid of flipped mask

	gauss_pyr_swap = gaussian_pyramid(swapped_image, depth) 
	laplace_pyr_swap = laplace_pyramid(gauss_pyr_swap) # laplace pyramid of face swapped image
	
	gauss_pyr_orig = gaussian_pyramid(image, depth) 
	laplace_pyr_orig = laplace_pyramid(gauss_pyr_orig) # laplace pyramid of original image

	# blend according to mask in each level
	blended_pyr = []
	for i in range(depth):
		blended_pyr.append(gauss_pyr_mask[i]*laplace_pyr_swap[depth-1-i] + gauss_pyr_mask_f[i]*laplace_pyr_orig[depth-1-i])# create the blended pyramid
	
	# collapse pyramid
	output = blended_pyr[depth-1]
	for i in range(len(blended_pyr)-1, 0, -1):
		expanded = cv2.pyrUp(output)
		output = expanded + blended_pyr[i - 1]

	return output
	
def gaussian_pyramid(image, depth):
	'''
	Used in laplace_blend()
	Using the pyrDown function iteratively to build a pyramid
	'''

	pyramid = [image]

	for _ in range(depth):
		image = cv2.pyrDown(image)
		pyramid.append(image)

	return pyramid


def laplace_pyramid(gauss_pyr):
	'''
	Used in laplace_blend()
	Iterate the gaussian pyramid
	'''
	depth = len(gauss_pyr)-1
	pyramid = [gauss_pyr[depth-1]]

	for i in range(depth-1, 0, -1):
		expanded_image = cv2.pyrUp(gauss_pyr[i])
		#print(gauss_pyr[i-1].shape)
		laplace = cv2.subtract(gauss_pyr[i-1], expanded_image)
		pyramid.append(laplace)
	   
	return pyramid

