import pandas as pd
from src.config import DATA_PATH

def load_data(filename):
    path = f"{DATA_PATH}/{filename}"
    df = pd.read_csv(path, parse_dates=True)
    return df
