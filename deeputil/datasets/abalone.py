import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/abalone.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 187 KB dataset with 4177 rows and 9 columns..........")
    assert len(pdObj) == 4177
    assert len(pdObj.columns) == 9
    return pdObj
