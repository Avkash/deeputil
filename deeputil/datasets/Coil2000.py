import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/coil_2000.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 2.8 MB dataset with 5822 and 86 columns..........")
    assert len(pdObj) == 5822
    assert len(pdObj.columns) == 86
    return pdObj
