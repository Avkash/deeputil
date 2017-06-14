import pandas as pd
import io
import requests


def load_data_as_pandas_df():
    dataset_url = "https://raw.githubusercontent.com/Avkash/mldl/master/data/adult_income_census.csv"
    streamData=requests.get(dataset_url).content
    pdObj=pd.read_csv(io.StringIO(streamData.decode('utf-8')))
    print("Downloaded 3.8 MB dataset with 32561 rows and 15 columns..........")
    assert len(pdObj) == 32561
    assert len(pdObj.columns) == 15
    return pdObj
