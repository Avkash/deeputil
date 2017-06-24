from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings
import pandas as pd
from keras.preprocessing import image
from keras import backend as K
from matplotlib import pyplot as plt



def get_model_summary(model, show_info=True):
    """
    :param model:
    :return:
    """
    if show_info is True:
        print("Generating model summary for the model....")
    return model.summary()


def get_model_configuration(model, show_info=True):
    """
    :param model:
    :return:
    """
    if show_info is True:
        print("Generating model configuration for the model directly from Keras....")
    return model.get_config()

def get_model_weights(model, show_info=True):
    """
    :param model:
    :return:
    """
    if show_info is True:
        print("Generating model weights for every layer in the model from Keras....")
    return model.get_weights()

def get_model_layers_count(model, show_info=True):
    """
    This function returns the integer value of total layers in the given model
    :param model: keras model
    :param show_info: Boolean True or False to show message or not
    :return: Integer
    """
    if show_info is True:
        print("Calculating model layers...")
    result = len(model.layers)
    if result is 0:
        if show_info is True:
            print("Total layers in this model are : " + str(result))
    return result


def get_model_input_image_shape_info(model, show_info=True):
    """
    This function returns the tupple of input image size which is required by given model.
    :param model:
    :param show_info: Boolean True or False to show message or not
    :return:
    """
    assert model
    input_shape = tuple()
    try:
        if show_info is True:
            print("Now getting very first layer from model..")
        layer_info = model.layers[0].get_config()
    except Exception:
        print("Error: There are no layers in this model..")
        return input_shape

    input_shape = model.layers[0].input_shape
    if show_info is True:
        print('The layer Input Shape is ' + str(input_shape) + " so you would need to import image of this size.")
    return input_shape


def get_model_activation_obj(model, model_layer_id, img_array, show_info=True):
    """
    This function returns the activation object for the give layer id along with image to be classified
    :param model:
    :param model_layer_id:
    :param img_array:
    :return:
    """
    assert model
    assert model_layer_id >= 0
    assert model_layer_id <= len(model.layers)
    #assert img_array
    if show_info is True:
        print("Now getting activation object for the selected layer " + str(model_layer_id) + " from the given model...")
    activation_temp = K.function([model.layers[0].input, K.learning_phase()],[model.layers[model_layer_id].output,])
    if show_info is True:
        print("Activation object is collected successfully")
    return activation_temp([img_array, 0])


def get_model_layer_details_by_layerId(model, layerId, show_info=True):
    """
    This function generates detailed information about the given layer in the model
    :param model:
    :param layerId:
    :param show_info:
    :return:
    """
    layer_info = model.layers[layerId].get_config()
    # print layer_info
    layer_name = layer_info['name']
    try:
        temp = layer_info['filters']
        layer_filters = temp
    except Exception:
        if show_info is True:
            print("There are no filters in this layer")
        layer_filters = "NaN"
    print(layer_info)
    print('Layer Id : ' + str(layerId))
    print('Layer Name : ' + layer_name)
    print('Layer Type : ' + model.layers[layerId].__class__.__name__)
    print('Layer filters : ' + str(layer_filters))
    print('Layer Number : ' + str(layerId))
    print('Layer Input Shape : ' + str(model.layers[layerId].input_shape))
    print('Layer output shape :' + str(model.layers[layerId].output_shape))
    print('Num of Parameters :' + str(model.layers[layerId].count_params()))

def get_model_layers_details_all(model, show_info=True):
    """
    This function returns details about each layer
    :param model:
    :param show_info:
    :return: dataframe of all layers info (Check for empty dataframe)
    """

    result_df = pd.DataFrame.empty

    if show_info is True:
        print("Collecting layer count.....")
    layer_count = len(model.layers)
    if layer_count is 0:
        if show_info is True:
            print("Total layers in this model are : " + str(layer_count))
            return result_df

    cols = ["Id", "LayerName", "Type", "Filters", "InputShape", "OutputShape", "Activation", "ParameterCount"]
    result_df = pd.DataFrame(columns=cols, index=range(layer_count))

    for layerId in range(layer_count):
        layer_info = model.layers[layerId].get_config()
        # print layer_info
        layer_name = layer_info['name']
        try:
            temp = layer_info['filters']
            layer_filters = temp
        except Exception:
            layer_filters = 0

        try:
            temp = layer_info['activation']
            layer_activation = temp
        except Exception:
            # "There are no filters in this layer"
            layer_activation = "None"

        input_shape = model.layers[layerId].input_shape
        output_shape = model.layers[layerId].output_shape
        layer_param = model.layers[layerId].count_params()
        result_df.loc[layerId].Id = layerId
        result_df.loc[layerId].LayerName = layer_name
        result_df.loc[layerId].Type = model.layers[layerId].__class__.__name__
        result_df.loc[layerId].Filters = layer_filters
        result_df.loc[layerId].InputShape = input_shape
        result_df.loc[layerId].OutputShape = output_shape
        result_df.loc[layerId].Activation = layer_activation
        result_df.loc[layerId].ParameterCount = layer_param
    if show_info is True:
        print("Total " + str(layer_count) + " layers details are collected.")
    return result_df

def get_model_layers_details_all_extended(model, image_array, show_info=True):
    """
    This function returns extended details about each layer when image is supplied for classification
    :param model:
    :param show_info:
    :return: dataframe of all layers info (Check for empty dataframe)
    """


    # TODO assert image_array to make sure we have qualified image array to get activiation object
    #assert image_array

    result_df = pd.DataFrame.empty

    if show_info is True:
        print("Collecting layer count.....")
    layer_count = len(model.layers)
    if layer_count is 0:
        if show_info is True:
            print("Total layers in this model are : " + str(layer_count))
            return result_df


    cols = ["Id", "LayerName", "Type", "Filters", "InputShape", "OutputShape", "Activation",
                        "ParameterCount", "Act_Shape", "FeatureMapSize", "FeatureMapCount"]
    result_df = pd.DataFrame(columns=cols, index=range(layer_count))

    for layerId in range(layer_count):
        layer_info = model.layers[layerId].get_config()
        # print layer_info
        layer_name = layer_info['name']
        try:
            temp = layer_info['filters']
            layer_filters = temp
        except Exception:
            layer_filters = "NaN"

        try:
            temp = layer_info['activation']
            layer_activation = temp
        except Exception:
            # "There are no filters in this layer"
            layer_activation = "None"

        input_shape = model.layers[layerId].input_shape
        output_shape = model.layers[layerId].output_shape
        layer_param = model.layers[layerId].count_params()
        result_df.loc[layerId].Id = layerId
        result_df.loc[layerId].LayerName = layer_name
        result_df.loc[layerId].Type = model.layers[layerId].__class__.__name__
        result_df.loc[layerId].Filters = layer_filters
        result_df.loc[layerId].InputShape = input_shape
        result_df.loc[layerId].OutputShape = output_shape
        result_df.loc[layerId].Activation = layer_activation
        result_df.loc[layerId].ParameterCount = layer_param

        local_activation_object = get_model_activation_obj(model, layerId, image_array, show_info=show_info)
        output_shape = np.shape(local_activation_object[0])
        result_df.loc[layerId].Act_Shape = output_shape

        temp = "NaN"
        try:
            if (np.shape(local_activation_object[0][0][0])):
                temp = (np.shape(local_activation_object[0][0][0]))
        except IndexError:
            if show_info is True:
                print("Error: Unable to get feature map count from the selected layer" + str(layerId) + " in this model..")
        result_df.loc[layerId].FeatureMapSize = temp
        #featuremap_size = np.shape(local_activation_object[0][0][0])
        #print('Featuremap Size : ' + str(featuremap_size))

        temp = "NaN"
        try:
            if (np.shape(local_activation_object[0][0][0]))[1]:
                temp = (np.shape(local_activation_object[0][0][0]))[1]
        except IndexError:
            if show_info is True:
                print("Error: Unable to get feature map count from the selected layer" + str(layerId) + " in this model..")
        result_df.loc[layerId].FeatureMapCount = temp
        #featuremaps_num = (np.shape(local_activation_object[0][0][0]))[1]
        #print('Featuremaps Count: ' + str(featuremaps_num))

    if show_info is True:
        print("Total " + str(layer_count) + " layers details are collected.")
    return result_df

def get_model_layer_feature_map_counts(model, model_layer_id, img_array, show_info=True):
    """
    :param model:
    :param model_layer_id:
    :param img_array:
    :return:
    """
    assert model
    assert model_layer_id >= -1
    result = 0
    if show_info is True:
        print("Now collecting feature map for the selected layer in the given model..")
    activation_temp = K.function([model.layers[0].input, K.learning_phase()], [model.layers[model_layer_id].output, ])
    activationObj = activation_temp([img_array, 0])
    try:
        if (np.shape(activationObj[0][0][0]))[1]:
            result = (np.shape(activationObj[0][0][0]))[1]
    except IndexError:
        if show_info is True:
            print("Error: Unable to get feature map from the selected layer in this model..")
    return result

def get_feature_map_counts_for_all_layers(model, img_array, show_info=True):
    """
    :param model:
    :param model_layer_id:
    :param img_array:
    :return:
    """
    assert model
    if show_info is True:
        print("Now collecting feature maps for all layers in the given model..")

    result_df = pd.DataFrame.empty

    layer_count = len(model.layers)
    if layer_count is 0:
        print("Error: This model has 0 layers")

    cols = ["LayerId", "FeatureMapCount"]
    result_df = pd.DataFrame(columns=cols, index=range(layer_count))

    for model_layer_id in range(layer_count):
        activation_temp = K.function([model.layers[0].input, K.learning_phase()], [model.layers[model_layer_id].output, ])
        activationObj = activation_temp([img_array, 0])
        result = "NaN"
        try:
            if (np.shape(activationObj[0][0][0]))[1]:
                result = (np.shape(activationObj[0][0][0]))[1]
        except IndexError:
            result = "NaN"

        result_df.loc[model_layer_id].LayerId = model_layer_id
        result_df.loc[model_layer_id].FeatureMapCount = result
    if show_info is True:
        print("Feature maps for all layers are collected.")

    return result_df

def display_full_feature_map_for_selected_layer_in_model(model, model_layer_id, img_array, showOnly=0, show_info=True):
    """
    This function generate feature map graph for each feature in the given model for selected layer
    :param model:
    :param layer_id:
    :param show_info:
    :return:
    """

    if model_layer_id > len(model.layers):
        print("Error: The given layer id is incorrect and higher then total layers in the given model.")
        return

    if model_layer_id < 0:
        print("Error: The given layer id is incorrect and lower then total layers in the given model.")
        return

    if show_info is True:
        print("Now collecting feature map for the selected layer in the given model..")
    activation_temp = K.function([model.layers[0].input, K.learning_phase()], [model.layers[model_layer_id].output, ])
    activationObj = activation_temp([img_array, 0])

    feature_map_total = 0
    try:
        if (np.shape(activationObj[0][0]))[2]:
            feature_map_total = (np.shape(activationObj[0][0][0]))[1]
    except IndexError:
        print("Error: Layer does not have any feature maps...")
        return

    if showOnly is 0:
        showOnly = feature_map_total

    if showOnly > feature_map_total:
        showOnly = feature_map_total

    if show_info is True:
        print("Now displaying " + str(feature_map_total) + " feature maps for the selected layer in the given model..")

    output_shape = np.shape(activationObj[0])
    if len(output_shape) == 2:
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(activationObj[0].T,cmap='gray')
        plt.show()
    else:
        fig = plt.figure(figsize=(16, 16))
        subplot_num = int(np.ceil(np.sqrt(feature_map_total)))
        if show_info is True:
            print("Plotting " + str(subplot_num) + " feature maps in each row.")
        for i in range(int(showOnly)):
            loc_fmap = fig.add_subplot(subplot_num, subplot_num, i+1)
            activationObj_local = activationObj[0]
            activationObj_local = activationObj_local[0]
            img_local = activationObj_local[0:, 0:, i]
            loc_fmap.imshow(img_local)
        plt.show()
    if show_info is True:
        print("You are watching total " + str(feature_map_total) + " features from layer #" + str(model_layer_id) + ".")


def display_individual_feature_for_selected_layer_in_model(model, model_layer_id, feature_map_id, img_array, show_info=True):
    """

    :param model:
    :param model_layer_id:
    :param img_array:
    :param show_info:
    :return:
    """
    if model_layer_id > len(model.layers):
        print("Error: The given layer id is incorrect and higher then total layers in the given model.")
        return

    if model_layer_id < 0:
        print("Error: The given layer id is incorrect and lower then total layers in the given model.")
        return

    if show_info is True:
        print("Now collecting feature map for the selected layer in the given model..")
    activation_temp = K.function([model.layers[0].input, K.learning_phase()], [model.layers[model_layer_id].output, ])
    activationObj = activation_temp([img_array, 0])

    feature_map_total = 0
    try:
        if (np.shape(activationObj[0][0]))[2]:
            feature_map_total = (np.shape(activationObj[0][0][0]))[1]
    except IndexError:
        print("Error: Layer does not have any feature maps...")
        return

    if feature_map_id > feature_map_total-1:
        print("Error: The given feature map id is incorrect and higher then total maps in the given layer.")
        return

    fig=plt.figure(figsize=(8,8))
    activationObj_local = activationObj[0]
    activationObj_local = activationObj_local[0]
    im_temp = activationObj_local[0:,0:,feature_map_id]
    plt.imshow(im_temp)
    plt.show()

    if show_info is True:
        print("You are watching feature map #" + str(feature_map_id) + " from the layer #" + str(model_layer_id) + ".")
