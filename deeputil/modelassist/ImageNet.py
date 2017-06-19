from __future__ import absolute_import
from __future__ import print_function

import warnings

import json
import pandas as pd
from ..utils.data_utils import get_file

CLASS_INDEX = None
IMAGENET_CLASS_JSON = "https://raw.githubusercontent.com/Avkash/mldl/master/data/imagenet/imagenet_class_index.json"

def get_imagenet_classes_as_df():
    global CLASS_INDEX
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         IMAGENET_CLASS_JSON,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))

    results = pd.DataFrame.empty

    class_count = len(CLASS_INDEX)
    if class_count == 0:
        return results

    cols = ["Index", "Id", "Class"]
    results = pd.DataFrame(columns=cols, index=range(class_count))
    for a in range(class_count):
        results.loc[a].Index = a
        results.loc[a].Id = CLASS_INDEX[str(a)][0]
        results.loc[a].Class = CLASS_INDEX[str(a)][1]

    return results

# End of file