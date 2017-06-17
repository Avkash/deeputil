from __future__ import absolute_import
from __future__ import print_function

from . import vgg16
from . import vgg19

def get_vgg16_model(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    print("The VGG16 model size is about ~0.5 GB will be downloaded now so make sure you have time and space to make it happen....")
    return vgg16.vgg16(include_top,weights,input_tensor,input_shape,pooling,classes)

def get_vgg19_model(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None,
                        classes=1000):
    print("The VGG19 model size is about ~0.5 GB will be downloaded now so make sure you have time and space to make it happen....")
    return vgg19.vgg19(include_top,weights,input_tensor,input_shape,pooling,classes)
