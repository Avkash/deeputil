import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/titanic_list.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 106 KB dataset with 1309 rows and 14 columns..........")
    assert len(pdObj) == 1309
    assert len(pdObj.columns) == 14
    return pdObj
