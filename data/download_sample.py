import pandas as pd


def download_sample(output_path: str, n_rows: int = 10000):
    url = "https://data.cityofchicago.org/api/views/wrvz-psew/rows.csv?accessType=DOWNLOAD"
    df = pd.read_csv(url, nrows=n_rows)
    df.to_csv(output_path, index=False)
