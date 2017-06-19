from __future__ import absolute_import
from __future__ import print_function

from . import VGG16
from . import VGG19
from . import InceptionV3
from . import Xception
from . import ResNet50

def get_vgg16_model(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    print("VGG16 model is about ~0.5GB, so make sure you have time and space to make it happen....")
    return VGG16.vgg16(include_top, weights, input_tensor, input_shape, pooling, classes)

def get_vgg19_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000):
    print("VGG19 model is about ~0.5GB, so make sure you have time and space to make it happen....")
    return VGG19.vgg19(include_top, weights, input_tensor, input_shape, pooling, classes)

def get_inceptionV3_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000):
    print("Inception V3 model is about ~0.5GB, so make sure you have time and space to make it happen....")
    return InceptionV3.InceptionV3(include_top, weights, input_tensor, input_shape, pooling, classes)

def get_xception_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000):
    print("Xception model is about ~0.5GB, so make sure you have time and space to make it happen....")
    return Xception.Xception(include_top, weights, input_tensor, input_shape, pooling, classes)

def get_resnet50_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000):
    print("ResNet50 model is about ~0.5GB, so make sure you have time and space to make it happen....")
    return ResNet50.ResNet50(include_top, weights, input_tensor, input_shape, pooling, classes)


