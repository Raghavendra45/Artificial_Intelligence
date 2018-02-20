from __future__ import division
import numpy as np
from matplotlib import image
import matplotlib.cm as cm
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

"""Helper image-processing code."""


def image_to_matrix(image_file, grays=False):
    """
    Convert .png image to matrix
    of values.
    params:
    image_file = str
    grays = Boolean
    returns:
    img = (color) np.ndarray[np.ndarray[np.ndarray[float]]]
    or (grayscale) np.ndarray[np.ndarray[float]]
    """
    img = image.imread(image_file)
    # in case of transparency values
    if len(img.shape) == 3 and img.shape[2] > 3:
        height, width, depth = img.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r, c, :] = img[r, c, 0:3]
        img = np.copy(new_img)
    if grays and len(img.shape) == 3:
        height, width = img.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r, c] = img[r, c, 0]
        img = new_img
    return img


def matrix_to_image(image_matrix, image_file):
    """
    Convert matrix of color/grayscale
    values  to .png image
    and save to file.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    image_file = str
    """
    # provide cmap to grayscale images
    c_map = None
    if len(image_matrix.shape) < 3:
        c_map = cm.Greys_r
    image.imsave(image_file, image_matrix, cmap=c_map)


def flatten_image_matrix(image_matrix):
    """
    Flatten image matrix from
    Height by Width by Depth
    to (Height*Width) by Depth
    matrix.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]

    returns:
    flattened_values = (color) numpy.ndarray[numpy.ndarray[float]]
    or (grayscale) numpy.ndarray[float]
    """
    if(len(image_matrix.shape) == 3):
        height, width, depth = image_matrix.shape
    else:
        height, width = image_matrix.shape
        depth = 1
    flattened_values = np.zeros([height*width,depth])
    for i, r in enumerate(image_matrix):
        for j, c in enumerate(r):
            flattened_values[i*width+j,:] = c
    return flattened_values


def unflatten_image_matrix(image_matrix, width):
    """
    Unflatten image matrix from
    (Height*Width) by Depth to
    Height by Width by Depth matrix.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[float]]
    or (grayscale) numpy.ndarray[float]
    width = int

    returns:
    unflattened_values =
        (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    """
    heightWidth = image_matrix.shape[0]
    height = int(heightWidth / width)
    if(len(image_matrix.shape) > 1):
        depth = image_matrix.shape[-1]
        unflattened_values = np.zeros([height, width, depth])
        for i in range(height):
            for j in range(width):
                unflattened_values[i,j,:] = image_matrix[i*width+j,:]
    else:
        depth = 1
        unflattened_values = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                unflattened_values[i,j] = image_matrix[i*width+j]
    return unflattened_values


def image_difference(image_values_1, image_values_2):
    """
    Calculate the total difference
    in values between two images.
    Assumes that both images have same
    shape.

    params:
    image_values_1 = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    image_values_2 = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]

    returns:
    dist = int
    """
    flat_vals_1 = flatten_image_matrix(image_values_1)
    flat_vals_2 = flatten_image_matrix(image_values_2)
    n, depth = flat_vals_1.shape
    dist = 0.
    point_thresh = 0.005
    for i in range(n):
        if depth > 1:
            new_dist = sum(abs(flat_vals_1[i] - flat_vals_2[i]))
            if new_dist > depth * point_thresh:
                dist += new_dist
        else:
            new_dist = abs(flat_vals_1[i] - flat_vals_2[i])
            if new_dist > point_thresh:
                dist += new_dist
    return dist
