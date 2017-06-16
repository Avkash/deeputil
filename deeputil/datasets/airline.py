import pandas as pd
import io
import requests


def load_airlines_base_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/airlines_base.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 1.3 MB dataset with 699 and 11 columns..........")
    assert len(pdObj) == 699
    assert len(pdObj.columns) == 11
    return pdObj


def load_airlines_1987_2008_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/airlines_1987_2008.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 4.4 MB dataset with 43978 and 31 columns..........")
    assert len(pdObj) == 43978
    assert len(pdObj.columns) == 31
    return pdObj
