from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings

from keras.preprocessing import image
from keras import backend as K
from matplotlib import pyplot as plt



def get_keras_model_summary(model):
    """
    :param model:
    :return:
    """
    print("Generating model summary for the model....")
    return model.summary()


def get_keras_model_configuration(model):
    """
    :param model:
    :return:
    """
    print("Generating model configuration for the model....")
    return model.get_config()

def get_keras_model_weights(model):
    """
    :param model:
    :return:
    """
    print("Generating model weights for every layer in the model....")
    return model.get_weights()

def get_keras_model_layers_count(model):
    """
    :param model:
    :return:
    """
    print("Calculating model layers.... Done!")
    return len(model.layers)


def get_keras_model_input_image_shape_info(model):
    """

    :param model:
    :return:
    """
    assert model
    try:
        # Getting very first layer from model
        layer_info = model.layers[0].get_config()
    except Exception:
        return "There are no layers in this model.."

    input_shape = model.layers[0].input_shape
    print('The layer Input Shape is ' + str(input_shape) + " so you would need to import image of this size.")
    return input_shape


def get_keras_model_activation_obj(model, model_layer_id, img_array):
    """

    :param model:
    :param model_layer_id:
    :param img_array:
    :return:
    """
    assert model
    assert model_layer_id >= 0
    #assert img_array
    activation_temp = K.function([model.layers[0].input, K.learning_phase()],[model.layers[model_layer_id].output,])
    return activation_temp([img_array,0])


def get_keras_model_layer_details_by_layerId(model, layerId):
    layer_info = model.layers[layerId].get_config()
    # print layer_info
    layer_name = layer_info['name']
    try:
        temp = layer_info['filters']
        layer_filters = temp
    except Exception:
        layer_filters = "There are no filters in this layer"
        # do nothing
    input_shape = model.layers[layerId].input_shape
    output_shape = model.layers[layerId].output_shape
    layer_param = model.layers[layerId].count_params()
    print('Layer Id : ' + str(layerId))
    print('Layer Name : ' + layer_name)
    print('Layer filters : ' + str(layer_filters))
    print('Layer Number : ' + str(layerId))
    print('Layer Input Shape : ' + str(input_shape))
    print('Layer output shape :' + str(output_shape))
    print('Num of Parameters :' + str(layer_param))

def get_keras_model_activation_details_by_layerId(activationObj, layerId):
    output_shape = np.shape(activationObj[0])
    featuremap_size = np.shape(activationObj[0][0][0])
    featuremaps_num = (np.shape(activationObj[0][0][0]))[1]
    print('Output Shape (activationId) : ' + str(output_shape))
    print('Featuremap Size : ' + str(featuremap_size))
    print('Featuremaps Count: ' + str(featuremaps_num))


def get_keras_model_layer_feature_map_counts(model, model_layer_id, img_array):
    """
    :param model:
    :param model_layer_id:
    :param img_array:
    :return:
    """
    assert model
    assert type(model_layer_id) == int
    assert model_layer_id >= -1
    activation_temp = K.function([model.layers[0].input, K.learning_phase()],[model.layers[model_layer_id].output,])
    activationObj = activation_temp([img_array,0])
    #output_shape = np.shape(activationObj[0])
    #featuremap_size = np.shape(activationObj[0][0][0])
    try:
        if (np.shape(activationObj[0][0][0]))[1]:
            return (np.shape(activationObj[0][0][0]))[1]
    except IndexError:
        return 0


def get_keras_model_layer_feature_maps(activationObj, showOnly):
    """

    :param activationObj:
    :param showOnly:
    :return:
    """
    assert showOnly > 0

    output_shape = np.shape(activationObj[0])

    try:
        if (np.shape(activationObj[0][0]))[2]:
            f_map_total = (np.shape(activationObj[0][0][0]))[1]
    except IndexError:
        return "Layer does not have any feature maps..."

    if (showOnly > f_map_total):
        showOnly == f_map_total

    print("Now showing " + str(f_map_total) + " features for this layer...")
    if len(output_shape)==2:
        fig=plt.figure(figsize=(16,16))
        plt.imshow(activationObj[0].T,cmap='gray')
        plt.show()
        #plt.savefig("featuremaps-layer-{}".format(self.layer) + '.png')
    else:
        fig=plt.figure(figsize=(16,16))
        subplot_num=int(np.ceil(np.sqrt(f_map_total)))
        print("Subplot Num : " + str(subplot_num))
        if showOnly <= 0:
            showOnly = f_map_total
        for i in range(int(showOnly)):
            ax = fig.add_subplot(subplot_num, subplot_num, i+1)
            #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
            activationObj_local = activationObj[0]
            activationObj_local = activationObj_local[0]
            im_t = activationObj_local[0:,0:,i]
            ax.imshow(im_t) #,cmap='gray')
        plt.show()


def get_keras_model_layer_individual_feature(activationObj, featureMapNumber):
    """

    :param activationObj:
    :param featureMapNumber:
    :return:
    """
    assert featureMapNumber >= 0

    try:
        if (np.shape(activationObj[0][0]))[2]:
            f_map_total = (np.shape(activationObj[0][0][0]))[1]
    except IndexError:
        return "Layer does not have any feature maps..."

    if (featureMapNumber > f_map_total):
        print("The feature id is higher then total feature map, showing the last one - " + str(f_map_total))
        featureMapNumber = f_map_total-1

    fig=plt.figure(figsize=(8,8))
    activationObj_local = activationObj[0]
    activationObj_local = activationObj_local[0]
    im_temp = activationObj_local[0:,0:,featureMapNumber]
    ##print(im_temp.shape)
    plt.imshow(im_temp) # ,cmap = 'gray')
    plt.show()


