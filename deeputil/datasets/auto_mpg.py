import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/auto_mpg.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 14 KB dataset with 398 rows and 8 columns..........")
    assert len(pdObj) == 398
    assert len(pdObj.columns) == 8
    return pdObj

