#! /usr/bin/env python3
import cv2
import numpy as np


def get_affine(src, src_tri, dest_tri, size):
    '''
	:param src:           Source image (Cropped area containing the triangle area)
	:param src_tri:       Source triangle
	:param dest_tri:      Destination triangle
	:param size:          Size of bounding box of triangle

	:return dst:          Affine warped src image
	'''
    
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dest_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morph_affine(tri_1, tri_2, orig_img, morphed_image, debug=False):
    '''
	:param tri_1:           Triangle in face 1 (pixel coordinates)
	:param tri_2:           Corresponding triangle in face 2 (pixel coordinates)
	:param orig_img:        Original frame from camera.
	:param morphed_image:   Copy of orig_img, this image we morph
    :param debug:           Set True to run in "debug" mode

	:return None:
	'''

    x_1, y_1, w_1, h_1 = cv2.boundingRect(np.float32([tri_1]))
    x_2, y_2, w_2, h_2 = cv2.boundingRect(np.float32([tri_2]))
    # maybe i can just use a rotation rectangle instead of offset stuff ? couldnt figure it out
    # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

    # makes an offset to fit the rectangle boxes better to the triangles.
    offset_triangle_1 = []
    offset_triangle_2 = []
    for coords in tri_1:
        offset_triangle_1.append(((coords[0] - x_1), (coords[1] - y_1)))
    for coords in tri_2:
        offset_triangle_2.append(((coords[0] - x_2), (coords[1] - y_2)))

    mask = np.zeros((h_2, w_2, 3))
    cv2.fillConvexPoly(mask, np.int32(offset_triangle_2), (1.0, 1.0, 1.0), 16, 0)
    # Get the part we are mapping from  face to face
    img1_within_boundary = orig_img[y_1:y_1 + h_1, x_1:x_1 + w_1]
    size_bounds_triangle_2 = (w_2, h_2)
    # make sure it is a triangle and not a line, just bug catching
    if 0 in img1_within_boundary.shape:
        return # Some error, just return

    transformed_area = get_affine(img1_within_boundary, offset_triangle_1, offset_triangle_2, size_bounds_triangle_2)
    # remove all parts of the transformed image outside the area we care about (triangle mask)
    transformed_triangle_only = transformed_area * mask
    shape1 = mask.shape
    shape2 = morphed_image[y_2:y_2 + h_2, x_2:x_2 + w_2].shape
    if shape1 != shape2:
        return # Some error, just return

    # slices tbe area we care about, adds the new face, and puts it back into where it belongs, back to mama
    morphed_image[y_2:y_2 + h_2, x_2:x_2 + w_2] = (morphed_image[y_2:y_2 + h_2, x_2:x_2 + w_2] * ((1.0, 1.0, 1.0) - mask)) + transformed_triangle_only
    if debug:
        cv2.imshow('Image during transformation', morphed_image)
        cv2.waitKey()
