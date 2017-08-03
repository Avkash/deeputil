from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b
from .. import model
from .. import utils
from .. import modelassist
from keras import backend
from PIL import Image


def getModelLayesFirst(is_top_included, image_style_tensor, imagenet_weights, show_info):
    interim_model = model.ModelsMaster.get_vgg16_model(include_top=is_top_included, input_tensor=image_style_tensor,
                                                       weights=imagenet_weights)
    layers = dict([(layer.name, layer.output) for layer in interim_model.layers])
    utils.helper_functions.show_print_message("Verifying all layers from the model:", show_info)
    p_layers = modelassist.KerasModelHelper.get_model_layers_details_all(interim_model)
    utils.helper_functions.show_print_message(p_layers, show_info)
    return layers


def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def style_loss(style, combination, height, width):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x, height, width):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


def eval_loss_and_grads(x, height, width, f_outputs):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        self.height = None
        self.width = None
        self.f_outputs = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x, self.height, self.width, self.f_outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def createNeuralDesign(height, width, input_image_location, style_image_location, iteration_count, show_info=True):
    loc_image = np.random.uniform(0, 255, (1, height, width, 3)) - 128.
    # Change the iteration count
    iterations = iteration_count

    # Lets get Input Images First
    utils.helper_functions.show_print_message("Now importing selected input image from the disk...", show_info)
    content_image_path = input_image_location
    content_image = Image.open(content_image_path)
    content_image = content_image.resize((height, width))

    # Display content image
    utils.helper_functions.show_print_message("Now converting input image into numpy array..", show_info)
    content_array = np.asarray(content_image, dtype='float32')
    content_array = np.expand_dims(content_array, axis=0)
    utils.helper_functions.show_print_message("The input image shape is " + str(content_array.shape) + ".", show_info)

    # TODO: Check for image channels and if there are over 3 generate error OR convert to 3 channels only

    # Lets get style image Second
    utils.helper_functions.show_print_message("Now importing selected style image from the disk...", show_info)
    style_image_path = style_image_location
    style_image = Image.open(style_image_path)
    style_image = style_image.resize((height, width))
    # Display content image
    # Show
    utils.helper_functions.show_print_message("Now converting style image into numpy array..", show_info)
    style_array = np.asarray(style_image, dtype='float32')
    style_array = np.expand_dims(style_array, axis=0)
    utils.helper_functions.show_print_message("The style image shape is " + str(style_array.shape) + ".", show_info)

    # For this, we need to perform two transformations:
    # [1] Subtract the mean RGB value (computed previously on the ImageNet training set and easily obtainable
    #     from Google searches) from each pixel.
    # [2] Flip the ordering of the multi-dimensional array from RGB to BGR (the ordering used in the paper).

    # Converting input image
    utils.helper_functions.show_print_message("Now transforming both style and input image numpy array as below:", show_info)
    utils.helper_functions.show_print_message("[1] Subtract the mean RGB value from each pixel", show_info)
    utils.helper_functions.show_print_message("[2] Flip the ordering of the multi-dimensional array from RGB to BGR", show_info)
    content_array[:, :, :, 0] -= 103.939
    content_array[:, :, :, 1] -= 116.779
    content_array[:, :, :, 2] -= 123.68
    content_array = content_array[:, :, :, ::-1]

    style_array[:, :, :, 0] -= 103.939
    style_array[:, :, :, 1] -= 116.779
    style_array[:, :, :, 2] -= 123.68
    style_array = style_array[:, :, :, ::-1]
    utils.helper_functions.show_print_message("Both style and input image numpy array are transformed:", show_info)

    # Combining both input and style images
    # Note We have reused content_image and style_image variable..
    content_image = backend.variable(content_array)
    style_image = backend.variable(style_array)
    combination_image = backend.placeholder((1, height, width, 3))

    # Convert both the images into tensorflow tensor
    utils.helper_functions.show_print_message("Now combing both style and input image array into a tensor.", show_info)
    input_tensor = backend.concatenate([content_image,
                                        style_image,
                                        combination_image], axis=0)
    # Lets get the model next
    utils.helper_functions.show_print_message("Now getting VGG16 Keras model without flatten layers..", show_info)
    vgg16_layers = getModelLayesFirst(is_top_included=False, imagenet_weights="imagenet", image_style_tensor=input_tensor, show_info=show_info)

    # Now use Layers from the above model

    # Configuration Settings
    utils.helper_functions.show_print_message("Setting default configuration for the style transfer", show_info)
    content_weight = 0.025
    style_weight = 5.0
    total_variation_weight = 1.0

    #  Now initialising  total loss to 0 and adding to it in stages.
    loss = backend.variable(0.)

    # [1] First working with Content loss
    utils.helper_functions.show_print_message("Now working on content loss function", show_info)
    layer_features = vgg16_layers['block2_conv2']
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss += content_weight * content_loss(content_image_features,combination_features)

    # [2] Now working with Style loss
    utils.helper_functions.show_print_message("Now working on style loss function", show_info)
    feature_layers = ['block1_conv2', 'block2_conv2',
                      'block3_conv3', 'block4_conv3',
                      'block5_conv3']

    for layer_name in feature_layers:
        layer_features = vgg16_layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features, height, width)
        loss += (style_weight / len(feature_layers)) * sl

    # [3] Now calculating total variation loss
    utils.helper_functions.show_print_message("Now calculating total variation loss", show_info)
    loss += total_variation_weight * total_variation_loss(combination_image, height, width)

    # [4] Now we need to define needed gradients and solve the optimisation problem
    utils.helper_functions.show_print_message("Now defining gradients solve the optimisation problem", show_info)
    grads = backend.gradients(loss, combination_image)

    outputs = [loss]
    outputs += grads
    f_outputs = backend.function([combination_image], outputs)

    # Initializing Evaluator class
    utils.helper_functions.show_print_message("Now Initializing Evaluator class and setting defaults", show_info)
    evaluator = Evaluator()
    evaluator.height = height
    evaluator.width = width
    evaluator.f_outputs = f_outputs
    utils.helper_functions.show_print_message("Now starting the style transfer loop", show_info)
    for i in range(iterations):
        utils.helper_functions.show_print_message('Iteration count: [' + str(i) + "]", show_info)
        start_time = time.time()
        loc_image, min_val, info = fmin_l_bfgs_b(evaluator.loss, loc_image.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        utils.helper_functions.show_print_message('Current loss value: [' + str(min_val) + "]", show_info)
        end_time = time.time()
        utils.helper_functions.show_print_message('Iteration %d completed in %ds' % (i, end_time - start_time), show_info)

    return loc_image


def processStyledImageForView(x, height, width, show_info):
    """

    :param x:
    :param height:
    :param width:
    :param show_info:
    :return:
    """
    utils.helper_functions.show_print_message("Now processing input styles image, ready to view!", show_info)
    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    utils.helper_functions.show_print_message("Done!!!", show_info)
    return x

