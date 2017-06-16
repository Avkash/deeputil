import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/winequality.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 314 KB dataset with 4898 rows and 12 columns..........")
    assert len(pdObj) == 4898
    assert len(pdObj.columns) == 12
    return pdObj
