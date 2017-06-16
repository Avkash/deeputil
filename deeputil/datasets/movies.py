import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/movies.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 453 KB dataset with 3883 and 18 columns..........")
    assert len(pdObj) == 3883
    assert len(pdObj.columns) == 18
    return pdObj

