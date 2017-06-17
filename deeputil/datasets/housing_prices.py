import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/housing_prices.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 450 KB dataset with 1460 and 81 columns..........")
    assert len(pdObj) == 1460
    assert len(pdObj.columns) == 81
    return pdObj
