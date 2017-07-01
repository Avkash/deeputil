from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from matplotlib import pyplot as plt
from .. import utils

def import_image_from_disk(imagePath, imageTargetSize, isGray= False, show_info=True):
    """
    :param imagePath:
    :param imageTargetSize:
    :return:
    """
    assert imagePath
    assert imageTargetSize

    utils.helper_functions.show_print_message("Now importing selected image from the disk...", show_info)
    img = image.load_img(imagePath, target_size=imageTargetSize, grayscale=isGray)
    return img

def is_image_gray(userImage, show_info=True):
    """
    :param image:
    :return:
    """
    assert userImage
    img_array = image.img_to_array(userImage)
    assert len(img_array.shape) == 3
    if (len(img_array.shape)) != 3:
        raise ValueError('The given image is not a valid image...')

    if (img_array.shape[2]) == 1:
        utils.helper_functions.show_print_message("Yes, the given image is gray", show_info)
        return True
    else:
        utils.helper_functions.show_print_message("No, the given image is NOT gray", show_info)
        return False


def convert_image_array(userImage, show_info=True):
    """
    :param image:
    :return:
    """
    assert userImage
    img_array = image.img_to_array(userImage)
    assert len(img_array.shape) == 3
    utils.helper_functions.show_print_message("Image as array shape:", show_info)
    return img_array

def preprocess_image_array(imgArray, show_info=True):
    """
    :param image:
    :return:
    """
    assert len(imgArray.shape) == 3

    if (imgArray.shape[2]) == 1:
        raise ValueError('Error: Preprocessing id done for color image only and input image is gray...')
    utils.helper_functions.show_print_message("Now pre-processing the image to get ready for classification..", show_info)
    img_array = np.expand_dims(imgArray, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def display_image_from_disk(image_path, target_size=(256, 256), show_info=True):
    """
    This function uses matplotlib to show an image, by loading it from disk
    :param image_path:
    :param show_info:
    :return: Display image as matplotlib graph
    """
    assert image_path
    utils.helper_functions.show_print_message("Now importing selected image from the disk with default size " + str(target_size) + " ...", show_info)
    img = image.load_img(image_path, target_size)
    plt.imshow(img)
    plt.show()


