from __future__ import absolute_import
from __future__ import print_function

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import warnings


def import_image_from_disk(imagePath, imageTargetSize, isGray= False):
    """
    :param imagePath:
    :param imageTargetSize:
    :return:
    """
    assert imagePath
    assert imageTargetSize
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


def convert_image_array(userImage):
    """
    :param image:
    :return:
    """
    assert userImage
    img_array = image.img_to_array(userImage)
    assert len(img_array.shape) == 3
    print('Image as array shape:', img_array.shape)
    return img_array

def preprocess_image_array(imgArray):
    """
    :param image:
    :return:
    """
    assert len(imgArray.shape) == 3

    if (imgArray.shape[2]) == 1:
        raise ValueError('Preprocessing id done for color image only and input image is gray...')

    img_array = np.expand_dims(imgArray, axis=0)
    img_array = preprocess_input(img_array)
    return img_array