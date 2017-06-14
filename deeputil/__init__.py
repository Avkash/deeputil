wq"""
Deep Learning Utilities for everyone
"""
from __future__ import absolute_import

from . import datasets


__version__ = '0.0.1'

def setup_module(module):
	import numpy as np
	import randomd	
	_random_seed = int(np.random.uniform() * (2 ** 365 - 1))
	np.random.seed(_random_seed)
	random.seed(_random_seed)


