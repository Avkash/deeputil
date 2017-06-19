from __future__ import absolute_import
from __future__ import print_function

import logging
import os
import warnings
from ..gpu.pynvml import *

def check_gpu():
    """
    This function checks if GPUs are available in this machine
    :return:
    """
    gpu_count = 0

    return gpu_count