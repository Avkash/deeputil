from __future__ import absolute_import
from __future__ import print_function

from keras import optimizers
from keras import metrics
from keras import losses

"""
keras_optimizers = [optimizers.Adadelta, optimizers.Adagrad, optimizers.adagrad,
                    optimizers.Adam, optimizers.TFOptimizer, optimizers.adadelta,
                    optimizers.rmsprop, optimizers.Nadam, optimizers.nadam
                    ]

keras_losses = [losses.categorical_crossentropy, losses.binary_crossentropy]

keras_metrics = [metrics.binary_accuracy, metrics.categorical_accuracy, metrics.cosine,
                 metrics.deserialize, metrics.MAE,
                 metrics.MAPE,
                 metrics.mape,
                 metrics.MSLE,
                 metrics.msle]
"""

keras_losses = ["categorical_crossentropy" , "sparse_categorical_crossentropy","binary_crossentropy",
                "mean_squared_error", "mean_absolute_error",
                "kullback_leibler_divergence", "poisson", "cosine_proximity",
                "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge",
                "categorical_hinge", "logcosh"]

keras_optimizers = ["RMSprop", "rmsprop", "SGD", "sgd", "Adagrad" , "adagrad", "Adadelta", "adadelta", "Adam", "adam",
                    "Adamax", "adamax", "Nadam", "nadam", "TFOptimizer", "tfoptimizer"]

keras_metrics = ["acc", "mae", "MAE", "msle", "MSLE", "mape", "MAPE", "cosine"]
