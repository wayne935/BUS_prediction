
import pandas as pd
import os

data_dir = '/Users/lufor/Documents/Program/EJ/bus_LSTM/dataset'
files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]

for file in files:
    file_path = os.path.join(data_dir, file)
    print(f"--- {file} ---")
    try:
        df = pd.read_excel(file_path, nrows=5)
        print(df.head())
        print(df.columns)
        print(df.index)
    except Exception as e:
        print(f"Error reading {file}: {e}")
