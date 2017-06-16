import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/us_communities.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 1.0 MB dataset with 1994 and 127 columns..........")
    assert len(pdObj) == 1994
    assert len(pdObj.columns) == 127
    return pdObj

