from __future__ import absolute_import
from __future__ import print_function

import warnings
from keras.models import model_from_json
from keras.models import Model
import keras
from .. import utils

from .. import definitions

def export_keras_model_as_h5(model, location, show_info=True):
    """

    :param model:
    :return:
    """
    assert isinstance(model, keras.engine.training.Model)
    model.save_weights(location)
    utils.helper_functions.show_print_message("Model (*.h5) is saved at " + location + "..", show_info)

def export_keras_model_as_HDF5(model, location, show_info=True):
    """

    :param model:
    :return:
    """
    assert isinstance(model, keras.engine.training.Model)
    model.save(location)
    utils.helper_functions.show_print_message("Model (*.HDF5) is saved at " + location + "..", show_info)


def export_keras_model_as_json(model, location, show_info=True):
    """

    :param model:
    :return:
    """
    assert isinstance(model, keras.engine.training.Model)
    model_json = model.to_json()
    with open(location, "w") as json_file:
        json_file.write(model_json)
    utils.helper_functions.show_print_message("Model (*.json) is saved at " + location + "..", show_info)


def import_keras_model_json_from_disk(location, show_info=True):
    """
    :param location:
    :return: model
    """
    # load json and create model
    json_file = open(location, 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    utils.helper_functions.show_print_message("Loading model json from the disk done..", show_info)
    return loaded_model


def import_keras_model_weights_from_disk(model, location, show_info=True):
    """
    :param location:
    :return:
    """
    assert isinstance(model, keras.engine.training.Model)
    model.load_weights(location)
    utils.helper_functions.show_print_message("Loading model weights from the disk done.", show_info)
    return model

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

    #assert model_config
    #assert model_weights
    #assert sample_weight_mode
    #assert model_loss_weights

    # Check if given loss is part of keras.losses
    utils.helper_functions.show_print_message("Losses: " + model_loss, show_info)
    if model_loss not in definitions.Definitions.keras_losses:
        utils.helper_functions.show_print_message("Error: The given loss function is not a keras loss function.", show_info)
        return model_local

    # Check if given optimizer is part of keras.optimizer
    utils.helper_functions.show_print_message("Optimizers: " + model_optimizer, show_info)
    if model_optimizer not in definitions.Definitions.keras_optimizers:
        utils.helper_functions.show_print_message("Error: The given optimizer is not a keras optimizer.", show_info)
        return model_local

    # Check if given metrics is part of keras.metrics
    utils.helper_functions.show_print_message("Metrics: " + str(model_metrics), show_info)
    len(model_metrics)

    for i in range(len(model_metrics)):
        if model_metrics[i] not in definitions.Definitions.keras_metrics:
            utils.helper_functions.show_print_message("Error: The given metrics is not a keras metrics.", show_info)
            return model_local

    model_local = import_keras_model_json_from_disk(model_config, show_info)
    model_local = import_keras_model_weights_from_disk(model_local, model_weights, show_info)
    model_local.compile(loss=model_loss,
              optimizer=model_optimizer,
              metrics=model_metrics)
    utils.helper_functions.show_print_message("Model config and weight import is done along with compile!", show_info)
    return model_local