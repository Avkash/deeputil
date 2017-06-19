import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/citibike.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded  3.1 MB dataset with 20000 rows and 15 columns..........")
    assert len(pdObj) == 20000
    assert len(pdObj.columns) == 15
    return pdObj

