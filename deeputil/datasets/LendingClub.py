import pandas as pd
import io
import requests


def load_full_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/lendingclub_full.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 32 MB dataset with 42538 and 52 columns..........")
    assert len(pdObj) == 42538
    assert len(pdObj.columns) == 52
    return pdObj

def load_10K_records_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/lendingclub_10K.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 6.8 MB dataset with 10000 and 52 columns..........")
    assert len(pdObj) == 10000
    assert len(pdObj.columns) == 52
    return pdObj

