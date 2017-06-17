import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/airquality.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 2.8 KB dataset with 153 and 6 columns..........")
    assert len(pdObj) == 153
    assert len(pdObj.columns) == 6
    return pdObj
