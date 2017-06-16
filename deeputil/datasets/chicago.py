import pandas as pd
import io
import requests


def load_chicago_crimes_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/chicagoCrimes.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 2.3 MB dataset with 9999 and 22 columns..........")
    assert len(pdObj) == 9999
    assert len(pdObj.columns) == 22
    return pdObj


def load_chicago_census_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/chicagoCensus.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 3.8 KB dataset with 78 and 9 columns..........")
    assert len(pdObj) == 78
    assert len(pdObj.columns) == 9
    return pdObj
