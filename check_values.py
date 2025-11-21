
import pandas as pd
import os

data_dir = '/Users/lufor/Documents/Program/EJ/bus_LSTM/dataset'
files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]

unique_values = set()
for file in files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_excel(file_path)
    unique_values.update(df['Status'].unique())

print(f"Unique Status values: {unique_values}")
