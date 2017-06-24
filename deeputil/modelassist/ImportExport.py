from __future__ import absolute_import
from __future__ import print_function

import warnings
from keras.models import model_from_json
from keras.models import Model

from .. import definitions

def export_keras_model_as_h5(model, location, show_info=True):
    """

    :param model:
    :return:
    """
    assert model
    model.save_weights(location)
    if show_info is True:
        print("Model (*.h5) is saved at " + location + "..")

def export_keras_model_as_HDF5(model, location, show_info=True):
    """

    :param model:
    :return:
    """
    assert model
    model.save(location)
    if show_info is True:
        print("Model (*.HDF5) is saved at " + location + "..")


def export_keras_model_as_json(model, location, show_info=True):
    """

    :param model:
    :return:
    """
    assert model
    model_json = model.to_json()
    with open(location, "w") as json_file:
        json_file.write(model_json)
    if show_info is True:
        print("Model (*.json) is saved at " + location + "..")


def import_keras_model_json_from_disk(location, show_info=True):
    """
    :param location:
    :return: model
    """
    # load json and create model
    json_file = open(location, 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    if show_info is True:
        print("Loading model json from the disk done..")
    return loaded_model


def import_keras_model_weights_from_disk(jsonModel, location, show_info=True):
    """
    :param location:
    :return:
    """
    jsonModel.load_weights(location)
    if show_info is True:
        print("Loading model weights from the disk done.")
    return jsonModel

def import_keras_model_config_and_weight_and_compile(model_config, model_weights,
                                                     model_loss_weights="none",
                                                     sample_weight_mode="none",
                                                     model_loss="categorical_crossentropy",
                                                     model_optimizer="rmsprop",
                                                     model_metrics=["acc"],
                                                     show_info=True
                                                     ):
    """
    This function loads a model config and weights from disk and then compile it from given parameters
    model_config:
    model_weights:
    model_weights_mode:
    loss:
    optimizer:
    metrics:
    :return: model (Keras Model)
    """
    model_local = Model

    assert model_config
    assert model_weights
    assert sample_weight_mode
    assert model_loss_weights

    # Check if given loss is part of keras.losses
    if show_info is True:
        print("Losses: " + model_loss)
    if model_loss not in definitions.Definitions.keras_losses:
        if show_info is True:
            print("Error: The given loss function is not a keras loss function.")
        return model_local

    # Check if given optimizer is part of keras.optimizer
    if show_info is True:
        print("Optimizers: " + model_optimizer)
    if model_optimizer not in definitions.Definitions.keras_optimizers:
        if show_info is True:
            print("Error: The given optimizer is not a keras optimizer.")
        return model_local

    # Check if given metrics is part of keras.metrics
    if show_info is True:
        print("Metrics: " + str(model_metrics))
    len(model_metrics)
    for i in range(len(model_metrics)):
        if model_metrics[i] not in definitions.Definitions.keras_metrics:
            if show_info is True:
                print("Error: The given metrics is not a keras metrics.")
            return model_local

    model_local = import_keras_model_json_from_disk(model_config, show_info)
    model_local = import_keras_model_weights_from_disk(model_local, model_weights, show_info)
    model_local.compile(loss=model_loss,
              optimizer=model_optimizer,
              metrics=model_metrics)
    return model_local