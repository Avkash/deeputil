
from keras.models import Model
from keras.utils.data_utils import get_file
from keras import backend as K
from .. import modelassist
import os

CONFIG_PATH='https://raw.githubusercontent.com/Avkash/mldl/master/data/models/mnist_config_100.json'
WEIGHTS_PATH = 'https://github.com/Avkash/mldl/raw/master/data/models/mnist_weight_100.h5'


def MNIST2000(show_info=True):
        """
        This is pre-built MNIST model trained for 2000 epochs with 99.38% accuracy
        :param show_info:
        :return: model as Keras.Model
        """
        # Getting Config first
        config_path = get_file('mnist_config_100.json',
                                CONFIG_PATH,
                                cache_subdir='models')

        # Getting weights next
        weights_path = get_file('mnist_weight_100.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')

        config_found = False
        if os.path.isfile(config_path):
            config_found = True
        else:
            if show_info is True:
                print("Error: Unable to get the MNIST model configuration on disk..")

        weight_found = False
        if os.path.isfile(weights_path):
            weight_found = True
        else:
            if show_info is True:
                print("Error: Unable to get the MNIST model weights on disk..")

        if config_found is False and weight_found is False:
            if show_info is True:
                print("Error: Unable to get the MNIST model..")

        return modelassist.ImportExport.import_keras_model_config_and_weight_and_compile(config_path, weights_path, show_info)
