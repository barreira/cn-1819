import pandas as pd
from sklearn.preprocessing import StandardScaler
# import numpy as np

data = pd.read_csv('mammographic_masses.data.txt',
                   names=['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'],
                   na_values=['?'])

print(data.head())

data.describe()

data = data.drop(['BI-RADS'], axis=1)
data = data.dropna()

features = data.loc[:, data.columns != 'Severity']
target = data['Severity']
print(features)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print(scaled_features)
