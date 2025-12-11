import pandas as pd

def load_raw_data():
    return pd.read_csv("data/adult.csv", na_values="?", skipinitialspace=True)
