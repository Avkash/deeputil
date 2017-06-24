from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from matplotlib import pyplot as plt

def import_image_from_disk(imagePath, imageTargetSize, isGray= False, show_info=True):
    """
    :param imagePath:
    :param imageTargetSize:
    :return:
    """
    assert imagePath
    assert imageTargetSize
    if show_info is True:
        print("Now importing selected image from the disk...")
    img = image.load_img(imagePath, target_size=imageTargetSize, grayscale=isGray)
    return img

def is_image_gray(userImage):
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
        return True
    else:
        return False


def convert_image_array(userImage, show_info=True):
    """
    :param image:
    :return:
    """
    assert userImage
    img_array = image.img_to_array(userImage)
    assert len(img_array.shape) == 3
    if show_info is True:
        print('Image as array shape:', img_array.shape)
    return img_array

def preprocess_image_array(imgArray, show_info=True):
    """
    :param image:
    :return:
    """
    assert len(imgArray.shape) == 3

    if (imgArray.shape[2]) == 1:
        raise ValueError('Error: Preprocessing id done for color image only and input image is gray...')
    if show_info is True:
        print("Now preprocessing the image to get ready for classification..")
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
    if show_info is True:
        print("Now importing selected image from the disk with default size " + str(target_size) + " ...")
    img = image.load_img(image_path, target_size)
    plt.imshow(img)
    plt.show()


