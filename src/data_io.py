import pandas as pd

raw_data = pd.read_csv("data/raw/tennis-master-data.csv")

print(raw_data.head())
print(raw_data.info())