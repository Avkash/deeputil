from __future__ import absolute_import
from __future__ import print_function

import warnings
import pandas as pd
from .. import model
from keras.models import Model
from .. import modelassist
from .. import imageassist
from .. import predict


g_network_list = ["VGG16", "VGG19", "RESNET50", "INCEPTION_V3", "XCEPTION"]

def get_vgg19Model(show_info=True):
    _model = model.ModelsMaster.get_vgg19_model(include_top=True, weights='imagenet', show_info=show_info)
    return _model

def get_vgg16Model(show_info=True):
    _model = model.ModelsMaster.get_vgg16_model(include_top=True, weights='imagenet', show_info=show_info)
    return _model

def get_InveptionV3_Model(show_info=True):
    _model = model.ModelsMaster.get_inceptionV3_model(include_top=True, weights='imagenet', show_info=show_info)
    return _model

def get_xception_Model(show_info=True):
    _model = model.ModelsMaster.get_xception_model(include_top=True, weights='imagenet', show_info=show_info)
    return _model

def get_resnet50_model(show_info=True):
    _model = model.ModelsMaster.get_resnet50_model(include_top=True, weights='imagenet', show_info=show_info)
    return _model

def perform_image_classification_by_network(image_path, network_type, top_n_classes=5, show_info=True):
    """
    :param image_path:
    :param network_type:
    :return:
    """
    pred_df = pd.DataFrame.empty

    if isinstance(image_path, str) is not True:
        print("Error: image_path must of qualified image path")
        return pred_df

    network_type = network_type.upper()
    if network_type not in g_network_list:
        print("Error: Please select any of given network types: VGG16, VGG19, RESNET50, INCEPTION_V3, XCEPTION")
        return pred_df

    model_local = Model
    if show_info is True:
        print("Getting selected Image network with 1000 classes first..")
    if network_type=="VGG16":
        model_local=get_vgg16Model(show_info)
    elif network_type=="VGG19":
        model_local=get_vgg19Model(show_info)
    elif network_type=="INCEPTION_V3":
        model_local=get_InveptionV3_Model(show_info)
    elif network_type=="XCEPTION":
        model_local=get_xception_Model(show_info)
    elif network_type=="RESNET50":
        model_local=get_resnet50_model(show_info)
    else:
        print("Error This network type is not supported")
        return pred_df

    if show_info is True:
        print("Now collecting information about selected network model..")
    layer_count = modelassist.KerasModelHelper.get_model_layers_count(model_local, show_info)
    if layer_count < 0:
        print("Error: Model has 0 layers.. exiting.")
        return model_local
    else:
        if show_info is True:
            print("Info: Model has " + str(layer_count) + " layers..")

    if show_info is True:
        print("Now getting input image size from the network first layer..")
    input_info = modelassist.KerasModelHelper.get_model_input_image_shape_info(model_local, show_info)

    image_input_size = (224, 224)  # Default size
    if len(input_info) == 4:
        image_input_size = (input_info[1], input_info[2])
    elif len(input_info) == 3:
        image_input_size = (input_info[2], input_info[3])
    elif len(input_info) == 2:
        image_input_size = (input_info[1], input_info[2])

    if show_info is True:
        print("Model will need an image of type " + str(image_input_size) + " to perform classification")

    if show_info is True:
        print("Now importing image based on input layer size and transforming it according to network input layer")
    image_local = imageassist.ImageUtils.import_image_from_disk(image_path, image_input_size, isGray=False, show_info=show_info)
    image_array = imageassist.ImageUtils.convert_image_array(image_local, show_info=show_info)
    image_array = imageassist.ImageUtils.preprocess_image_array(image_array, show_info=show_info)

    if show_info is True:
        print("Now perform image classification based in selected network")
    pred_df = predict.Prediction.perform_image_classification_by_model(model_local, image_array,
                                                                       is_image_array=True,
                                                                       top_n_classes=top_n_classes,
                                                                       show_info=show_info)
    if show_info is True:
        print("Classification is done!!")
    return pred_df


