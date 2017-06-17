import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/seeds_with_header.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 12 KB dataset with 210 and 8 columns..........")
    assert len(pdObj) == 210
    assert len(pdObj.columns) == 8
    return pdObj
