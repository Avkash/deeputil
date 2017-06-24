import pandas as pd
import io
import requests


def load_emoji_base_data_as_pandas_df(show_info=True):
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/emoji_base.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    if show_info is True:
        print("Downloaded 91 KB dataset with 2389 and 2 columns..........")
    assert len(pdObj) == 2389
    assert len(pdObj.columns) == 2
    return pdObj


def load_emoji_extended_data_as_pandas_df(show_info=True):
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/emoji4updated.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    if show_info is True:
        print("Downloaded 192 KB dataset with 2389 and 4 columns..........")
    assert len(pdObj) == 2389
    assert len(pdObj.columns) == 4
    return pdObj
