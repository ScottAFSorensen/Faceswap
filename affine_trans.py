#! /usr/bin/env python3
import cv2
import numpy as np


def get_affine(src, src_tri, dest_tri, size):
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dest_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_affine(tri_1, tri_2, orig_img, morphed_image):
    x_1, y_1, w_1, h_1 = cv2.boundingRect(np.float32([tri_1]))
    x_2, y_2, w_2, h_2 = cv2.boundingRect(np.float32([tri_2]))
    # maybe i can just use a rotation rectangle instead of offset stuff ?
    # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

    # Offset points by left top corner of the respective rectangles
    offset_triangle_1 = []
    offset_triangle_2 = []
    # for the <x,y> coordinates of each point the triangle find the offset
    # move this into a separate function if you need to do it a for a lot of triangles
    for coords in tri_1:
        offset_triangle_1.append(((coords[0] - x_1), (coords[1] - y_1)))
    for coords in tri_2:
        offset_triangle_2.append(((coords[0] - x_2), (coords[1] - y_2)))

    mask = np.zeros((h_2, w_2, 3))
    cv2.fillConvexPoly(mask, np.int32(offset_triangle_2), (1.0, 1.0, 1.0))

    # Get the part we are mapping
    img1_within_boundary = orig_img[y_1:y_1 + h_1, x_1:x_1 + w_1]
    size_bounds_triangle_2 = (w_2, h_2)

    transformed_area = get_affine(img1_within_boundary, offset_triangle_1, offset_triangle_2, size_bounds_triangle_2)
    # remove all parts of the transformed image outside the area we care about (triangle mask)
    transformed_triangle_only = transformed_area * mask

    # Combine below into 1 line.
    # slice the current area out of the in the image we are mapping the face to
    morphed_image[y_2:y_2 + h_2, x_2:x_2 + w_2] = morphed_image[y_2:y_2 + h_2, x_2:x_2 + w_2] * ((1.0, 1.0, 1.0) - mask)
    # slice the transformed area back in its place
    morphed_image[y_2:y_2 + h_2, x_2:x_2 + w_2] = morphed_image[y_2:y_2 + h_2, x_2:x_2 + w_2] + transformed_triangle_only
    return morphed_image
