import pandas as pd

data = pd.read_csv('Data/HistoricalPrices.csv')

data = data.rename(columns={' Open': 'Open', ' High': 'High', ' Low': 'Low', ' Close': 'Close'})
print(data.head())