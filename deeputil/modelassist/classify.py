from __future__ import absolute_import
from __future__ import print_function

import warnings
from .. import model
from keras.models import Model
from .. import modelassist
from .. import imageassist

g_network_list = ["VGG16", "VGG19", "RESNET50", "INCEPTION_V3", "XCEPTION"]

def get_vgg19Model():
    _model = model.models_master.get_vgg19_model(include_top=True, weights='imagenet')
    return _model

def get_vgg16Model():
    _model = model.models_master.get_vgg16_model(include_top=True, weights='imagenet')
    return _model

#network_options = {'VGG16': get_vgg16Model(), 'VGG19': get_vgg19Model()}


def perform_image_classification_by_network(image_path, network_type):
    """
    :param image_path:
    :param network_type:
    :return:
    """
    if isinstance(image_path, str) != True:
        print("Note: image_path must of qualified image path")

    network_type = network_type.upper()
    if network_type not in g_network_list:
        print("Please select any of given network types: VGG16, VGG19, RESNET50, INCEPTION_V3, XCEPTION")

    model_local = Model
    ## Step 1.: We need to get network first
    if network_type == "VGG16":
        model_local = get_vgg16Model()
    elif network_type == "VGG19":
        model_local = get_vgg19Model()

    ## Get some info about the model
    layer_count = modelassist.keras_model_details.get_keras_model_layers_count(model_local)
    if layer_count < 0:
        print("Model has 0 layers.. exiting.")
        return model_local
    else:
        print("Model has " + str(layer_count) + " layers..")

    ## Step 2: Getting network input type from the first layer
    input_info = modelassist.keras_model_details.get_keras_model_input_image_shape_info(model_local)

    image_input_size = (224,224) # Default size
    if len(input_info) == 4:
        image_input_size = (input_info[1], input_info[2])
    elif len(input_info) == 3:
        image_input_size = (input_info[2], input_info[3])
    elif len(input_info) == 2:
        image_input_size = (input_info[1], input_info[2])

    print("Model will need an image of type " + str(image_input_size) + " to classify..")

    ## Step 3: import image based on input layer size
    image_local = imageassist.image_utils.import_image_from_disk(image_path, image_input_size, isGray=False)
    image_array = imageassist.image_utils.convert_image_array(image_local)
    image_array = imageassist.image_utils.preprocess_image_array(image_array)

    for i in range(layer_count):
        print(modelassist.keras_model_details.get_keras_model_layer_feature_map_counts(model_local, i, image_array))

    ## Step 4: Perform image classification
    # Check and get image
    return model_local


