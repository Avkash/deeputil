import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/arrhythmia.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 393 KB dataset with 451 rows and 280 columns..........")
    assert len(pdObj) == 451
    assert len(pdObj.columns) == 280
    return pdObj
