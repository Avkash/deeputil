import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/walking.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 183 KB dataset with 151 and 124 columns..........")
    assert len(pdObj) == 151
    assert len(pdObj.columns) == 124
    return pdObj
