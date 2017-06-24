from __future__ import absolute_import
from __future__ import print_function

from keras.applications.imagenet_utils import decode_predictions
from .. import imageassist
import json
import pandas as pd
from ..utils.data_utils import get_file

CLASS_INDEX = None
IMAGENET_CLASS_JSON = "https://raw.githubusercontent.com/Avkash/mldl/master/data/imagenet/imagenet_class_index.json"

def perform_image_classification_by_model(model, input_image, is_image_array = False, top_n_classes=5, show_info=True):
    """
    In this function we will classify an image based on the model given by the use.
    :param model: a Keras Model
    :param image_array:
    :return:
    """
    global CLASS_INDEX
    results = pd.DataFrame.empty
    if show_info is True:
        print("Starting process to generate classes probabilities now..")
    if is_image_array == True:
        preds = model.predict(input_image)
    else:
        input_image_array = imageassist.ImageUtils.convert_image_array(input_image)
        input_image_array = imageassist.ImageUtils.preprocess_image_array(input_image_array)
        preds = model.predict(input_image_array)

    if len(preds.shape) != 2:
        print("Error: The predictions values are not in the shape of tupple as (1,1000).")
        return results

    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         IMAGENET_CLASS_JSON,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))

    class_count = len(CLASS_INDEX)
    if class_count == 0:
        print("Error: There was some problem reading imagenet classes...")
        return results

    if top_n_classes > class_count:
        top_n_classes=class_count

    cols = ["ClassName", "ClassId", "Probability"]
    results = pd.DataFrame(columns=cols, index=range(top_n_classes))

    if show_info is True:
        print("Classification completed, now generating prediction dataframe ..")
    for pred in preds:
        # Getting top results in the prediction through index
        top_indices = pred.argsort()[-top_n_classes:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        for k in range(len(result)):
            results.loc[k].ClassName = result[k][1]
            results.loc[k].ClassId = result[k][0]
            results.loc[k].Probability = result[k][2]
    return results
