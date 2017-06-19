from __future__ import absolute_import
from __future__ import print_function

import warnings
from .. import model
from keras.models import Model
from .. import modelassist
from .. import imageassist
from .. import predict


g_network_list = ["VGG16", "VGG19", "RESNET50", "INCEPTION_V3", "XCEPTION"]

def get_vgg19Model():
    _model = model.ModelsMaster.get_vgg19_model(include_top=True, weights='imagenet')
    return _model

def get_vgg16Model():
    _model = model.ModelsMaster.get_vgg16_model(include_top=True, weights='imagenet')
    return _model

def get_InveptionV3_Model():
    _model = model.ModelsMaster.get_inceptionV3_model(include_top=True, weights='imagenet')
    return _model

def get_xception_Model():
    _model = model.ModelsMaster.get_xception_model(include_top=True, weights='imagenet')
    return _model

def get_resnet50_model():
    _model = model.ModelsMaster.get_resnet50_model(include_top=True, weights='imagenet')
    return _model


#network_options = {'VGG16': get_vgg16Model(), 'VGG19': get_vgg19Model()}

def perform_image_classification_by_network(image_path, network_type, top_n_classes=5):
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
    if network_type=="VGG16":
        model_local=get_vgg16Model()
    elif network_type=="VGG19":
        model_local=get_vgg19Model()
    elif network_type=="INCEPTION_V3":
        model_local=get_InveptionV3_Model()
    elif network_type=="XCEPTION":
        model_local=get_xception_Model()
    elif network_type=="RESNET50":
        model_local=get_resnet50_model()
    else:
        print("This network type is not supported")
        return model_local

    ## Get some info about the model
    layer_count = modelassist.KerasModelHelper.get_keras_model_layers_count(model_local)
    if layer_count < 0:
        print("Model has 0 layers.. exiting.")
        return model_local
    else:
        print("Model has " + str(layer_count) + " layers..")

    ## Step 2: Getting network input type from the first layer
    input_info = modelassist.KerasModelHelper.get_keras_model_input_image_shape_info(model_local)

    image_input_size = (224,224) # Default size
    if len(input_info) == 4:
        image_input_size = (input_info[1], input_info[2])
    elif len(input_info) == 3:
        image_input_size = (input_info[2], input_info[3])
    elif len(input_info) == 2:
        image_input_size = (input_info[1], input_info[2])

    print("Model will need an image of type " + str(image_input_size) + " to classify..")

    ## Step 3: import image based on input layer size
    image_local = imageassist.ImageUtils.import_image_from_disk(image_path, image_input_size, isGray=False)
    image_array = imageassist.ImageUtils.convert_image_array(image_local)
    image_array = imageassist.ImageUtils.preprocess_image_array(image_array)

    ## Step 4: Perform image classification
    pred_df = predict.Prediction.perform_image_classification_by_model(model_local, image_array,
                                                                       is_image_array=True,
                                                                       top_n_classes=top_n_classes)
    # Check and get image
    return pred_df


