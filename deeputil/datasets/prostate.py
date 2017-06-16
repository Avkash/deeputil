import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/prostate.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 9.0 KB dataset with 380 rows and 9 columns..........")
    assert len(pdObj) == 380
    assert len(pdObj.columns) == 9
    return pdObj
