import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


#################################################### PREPROCESSING #####################################################


# Read dataset from CSV file

df = pd.read_csv('mammographic_masses.data.txt',
                 names=['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'],
                 na_values=['?'])

# Drop unrequired feature 'BI-RADS'

df = df.drop(['BI-RADS'], axis=1)

# Discard lines with NaN values

df = df.dropna()

# Separate features from target

df_features = pd.DataFrame(df.drop('Severity', axis=1))
df_target = df['Severity']

# df_features = pd.DataFrame(df.drop('Severity', axis=1).values)  # .values used so that old indexes are discarded
# df_target = pd.Series(df['Severity'].values)  # .values used so that old indexes are discarded

# Standardize feature data

scaler = StandardScaler()
df_features[df_features.columns] = scaler.fit_transform(df_features[df_features.columns])  # scales the existing df


####################################################### MODEL ##########################################################


def create_model(hidden_layers=2, nodes_per_layer=3, activation_fn='relu', learning_rate=10e-2):
    model = Sequential()
    model.add(Dense(4, activation=activation_fn, input_shape=(4,)))  # input layer

    for i in range(hidden_layers):
        model.add(Dense(nodes_per_layer, activation=activation_fn))

    model.add(Dense(1, activation=activation_fn))  # output layer

    adam = Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

    return model


classifier = KerasClassifier(build_fn=create_model, verbose=0)

# hyperparameters = {
#     'hidden_layers': [2, 4, 8],
#     'nodes_per_layer': list(range(1, 21)),
#     'activation_fn': ['relu', 'sigmoid'],
#     'learning_rate': [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2]
# }

# Cross-validation (k = 10 folds) and Results

scores = cross_val_score(classifier, df_features, df_target, cv=10)
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
