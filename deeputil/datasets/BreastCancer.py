import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/breast_cancer.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 24 KB dataset with 699 and 11 columns..........")
    assert len(pdObj) == 699
    assert len(pdObj.columns) == 11
    return pdObj
