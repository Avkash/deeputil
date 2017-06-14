from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
#from . import  model_details

def get_model_summary(model):
    """
    :param model:
    :return:
    """
    print("Generating model summary for the model....")
    return model.summary()


def get_model_configuration(model):
    """
    :param model:
    :return:
    """
    print("Generating model configuration for the model....")
    return model.get_config()

