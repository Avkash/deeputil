import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/UCI_Credit_Card.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 2.7 MB dataset with 30000 and 25 columns..........")
    assert len(pdObj) == 30000
    assert len(pdObj.columns) == 25
    return pdObj
