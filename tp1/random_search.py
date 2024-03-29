import os
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from math import sqrt


def mkdir(dir_name):
    '''
    Cria pasta cujo path é recebido como parâmetro.
    :param dir_name: Path para a pasta a ser criada
    '''
    try:
        os.rmdir(dir_name)
    except OSError:
        pass
    try:
        os.mkdir(dir_name)
    except OSError:
        pass


#################################################### PREPROCESSING #####################################################


# Read dataset from CSV file

df = pd.read_csv('mammographic_masses.data.txt',
                 names=['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'],
                 na_values=['?'])

# Drop unrequired feature 'BI-RADS'

df = df.drop(['BI-RADS'], axis=1)

# Print dataset description to file

pd.options.display.max_columns = 2000
print(df.describe(), file=open("dataset_description.txt", 'w'))

# Generate feature distribution graphs to files inside dist folder

mkdir("dist")

nRows = df.shape[0]
nBins = int(round(sqrt(nRows)))  # binning

for key in df.keys():
    df.hist(column=key, bins=nBins)
    fig_name = "dist-" + key + ".png"
    plt.savefig("dist/" + fig_name)

# Discard lines with NaN values

df = df.dropna()

# Separate features from target

features = pd.DataFrame(df.drop('Severity', axis=1))
target = df['Severity']

# df_features = pd.DataFrame(df.drop('Severity', axis=1).values)  # .values used so that old indexes are discarded
# df_target = pd.Series(df['Severity'].values)  # .values used so that old indexes are discarded

# Standardize feature data

scaler = StandardScaler()
features[features.columns] = scaler.fit_transform(features[features.columns])  # scales the existing df


####################################################### MODEL ##########################################################


# Creates Keras model
def create_model(hidden_layers=2, nodes_per_layer=3, activation_fn='relu', learning_rate=1e-2):
    model = Sequential()
    model.add(Dense(4, activation=activation_fn, input_shape=(4,)))  # input layer

    for i in range(hidden_layers):
        model.add(Dense(nodes_per_layer, activation=activation_fn))

    model.add(Dense(1, activation=activation_fn))  # output layer

    adam = Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

    return model


# Prints the top-X (by default = 3) results from model's hyperparameters optimization
def print_top_results(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (+/- {1:.3f})".format(results['mean_test_score'][candidate],
                                                                        results['std_test_score'][candidate]))
            print("Parameters: {0}\n".format(results['params'][candidate]))


# Create Keras classifier model

classifier = KerasClassifier(build_fn=create_model, verbose=0)

# Hyperparameter distribution

hp_dist = {
    'hidden_layers': [2, 4, 8, 16, 32],
    'nodes_per_layer': list(range(1, 21)),
    'activation_fn': ['relu', 'sigmoid'],
    'learning_rate': np.random.uniform(low=1e-8, high=1e-2, size=(10,))
}

# Random Search to optimize hyperparameters + Results

random_search = RandomizedSearchCV(classifier, param_distributions=hp_dist, n_iter=20, cv=10)
random_search.fit(features, target)
print_top_results(random_search.cv_results_)
