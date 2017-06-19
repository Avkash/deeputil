import pandas as pd
import io
import requests

def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/iris.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 4.4 KB dataset with 149 rows and 5 columns..........")
    assert len(pdObj) == 149
    assert len(pdObj.columns) == 5
    return pdObj
