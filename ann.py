import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read dataset from CSV file

df = pd.read_csv('mammographic_masses.data.txt',
                 names=['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'],
                 na_values=['?'])

# Drop unrequired feature 'BI-RADS'

df = df.drop(['BI-RADS'], axis=1)

# Discard lines with NaN values

df = df.dropna()

# Create separate dataframes from features and target

df_features = pd.DataFrame(df.drop('Severity', axis=1).values)  # .values used so that old indexes are discarded
df_target = pd.DataFrame(df['Severity'].values)  # .values used so that old indexes are discarded

print(df_features.head())

# Standardize feature data

scaler = StandardScaler()
df_features[df_features.columns] = scaler.fit_transform(df_features[df_features.columns])  # scales the existing df

# Split data in training and testing datasets

X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3)

# Tensorflow

columns = [tf.feature_column.numeric_column(key="Age"),
           tf.feature_column.numeric_column(key="Shape"),
           tf.feature_column.numeric_column(key="Margin"),
           tf.feature_column.numeric_column(key="Density")]

classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], feature_columns=columns, n_classes=2)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=20, shuffle=True)

classifier.train(input_fn=input_func, steps=500)

# Model evaluation


