from __future__ import absolute_import
from __future__ import print_function

import warnings
from keras.models import model_from_json
from keras.models import Model

def export_keras_model_as_h5(model, location):
    """

    :param model:
    :return:
    """
    assert model
    model.save_weights(location)
    print("Model (*.h5) is saved at " + location + "..")

def export_keras_model_as_HDF5(model, location):
    """

    :param model:
    :return:
    """
    assert model
    model.save(location)
    print("Model (*.HDF5) is saved at " + location + "..")


def export_keras_model_as_json(model, location):
    """

    :param model:
    :return:
    """
    assert model
    model_json = model.to_json()
    with open(location, "w") as json_file:
        json_file.write(model_json)
    print("Model (*.json) is saved at " + location + "..")


def import_keras_model_json_from_disk(location):
    """
    :param location:
    :return: model
    """
    # load json and create model
    json_file = open(location, 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    print("Loading model json from the disk done..")
    return loaded_model


def import_keras_model_weights_from_disk(jsonModel, location):
    """
    :param location:
    :return:
    """
    jsonModel.load_weights(location)
    print("Loading model weights from the disk done. Your next step is to compile the model with appropriate settings.")
    return jsonModel

