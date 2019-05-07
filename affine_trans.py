#! /usr/bin/env python
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
    x:x+w, y:y+h
    x_2, y_2, w_2, h_2= cv2.boundingRect(np.float32([tri_2]))

    # Might have to do some offset shit.. don't understand quite
    # maybe i ca just use a rotation rectangle ? https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

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
    cv2.imshow(mask)
    cv2.fillConvexPoly(mask, np.int32(offset_triangle_2), (1.0, 1.0, 1.0))

    # Get the part we are mapping
    img1_within_boundary = orig_img[y_1:y_1 + h_1, x_1:x_1 + w_1]

    return image
