import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/higgs.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 8.6 MB dataset with 15000 and 29 columns..........")
    assert len(pdObj) == 15000
    assert len(pdObj.columns) == 29
    return pdObj

