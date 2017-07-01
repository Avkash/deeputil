from __future__ import absolute_import
from __future__ import print_function

from . import VGG16
from . import VGG19
from . import InceptionV3
from . import Xception
from . import ResNet50
from . import Mnist
from . import Cifar10
from .. import utils


def get_vgg16_model(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000,
                    show_info=True):
    utils.helper_functions.show_print_message("VGG16 model is ~0.5GB, make sure you have time and space...", show_info)
    return VGG16.vgg16(include_top, weights, input_tensor, input_shape, pooling, classes)


def get_vgg19_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000,
                    show_info=True):
    utils.helper_functions.show_print_message("VGG19 model is ~0.5GB, make sure you have time and space...", show_info)
    return VGG19.vgg19(include_top, weights, input_tensor, input_shape, pooling, classes)


def get_inceptionV3_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000,
                          show_info=True):
    utils.helper_functions.show_print_message("Inception V3 model is ~0.5GB, make sure you have time and space...", show_info)
    return InceptionV3.InceptionV3(include_top, weights, input_tensor, input_shape, pooling, classes)


def get_xception_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000,
                       show_info=True):
    utils.helper_functions.show_print_message("Xception model is ~0.5GB, make sure you have time and space...", show_info)
    return Xception.Xception(include_top, weights, input_tensor, input_shape, pooling, classes)


def get_resnet50_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000,
                       show_info=True):
    utils.helper_functions.show_print_message("ResNet50 model is ~0.5GB, make sure you have time and space...", show_info)
    return ResNet50.ResNet50(include_top, weights, input_tensor, input_shape, pooling, classes)


def get_mnist_model(show_info=True):
    utils.helper_functions.show_print_message("This pre-built MNIST model is ~20MB and downloaded from the internet.", show_info)
    return Mnist.MNIST2000(show_info)


def get_cifar10_model(show_info=True):
    utils.helper_functions.show_print_message("This pre-built CIFAR10 model is ~20MB and downloaded from the internet.", show_info)
    return Cifar10.CIFAR10(show_info)