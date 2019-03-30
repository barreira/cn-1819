import GPy, GPyOpt
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# MNIST class

class MM():
    def __init__(self, hidden_layers=2, nodes_per_layer=3, activation_fn=0, learning_rate=10e-2):
        self.hidden_layers = hidden_layers
        self.nodes_per_layer = nodes_per_layer
        if activation_fn == 0:
            self.activation_fn = 'relu'
        else:
            self.activation_fn = 'sigmoid'
        self.learning_rate = learning_rate
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.mm_data()
        self.__model = self.mm_model()

    # load mnist data from keras dataset
    def mm_data(self):
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

        # Standardize feature data

        scaler = StandardScaler()
        df_features[df_features.columns] = scaler.fit_transform(df_features[df_features.columns])

        x_train, x_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.33, random_state=42)
        return x_train, x_test, y_train, y_test

    # mnist model
    def mm_model(self):
        model = Sequential()
        model.add(Dense(4, activation=self.activation_fn, input_shape=(4,)))  # input layer

        for i in range(self.hidden_layers):
            model.add(Dense(self.nodes_per_layer, activation=self.activation_fn))

        model.add(Dense(1, activation=self.activation_fn))  # output layer

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

        return model

    # fit mnist model
    def mm_fit(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)

        self.__model.fit(self.__x_train, self.__y_train, verbose=0, callbacks=[early_stopping])

    # evaluate mnist model
    def mm_evaluate(self):
        self.mm_fit()

        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, verbose=0)
        return evaluation


# #### Runner function for the MNIST model

# function to run mnist class
def run_mm(hidden_layers=2, nodes_per_layer=3, activation_fn=0, learning_rate=10e-2):
    _mnist = MM(hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer, activation_fn=activation_fn,
                learning_rate=learning_rate)
    mnist_evaluation = _mnist.mm_evaluate()
    return mnist_evaluation


# ## Bayesian Optimization
# #### bounds for hyper parameters

# bounds for hyper-parameters in mnist model
# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'hidden_layers', 'type': 'discrete', 'domain': (2, 4, 8)},
          {'name': 'nodes_per_layer', 'type': 'discrete', 'domain': tuple(range(1, 21))},
          {'name': 'activation_fn', 'type': 'discrete', 'domain': (0, 1)},
          {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-8, 1e-2)}]


# #### Bayesian Optimization

# function to optimize mnist model
def f(x):
    print(x)
    evaluation = run_mm(
        hidden_layers=int(x[:, 0]),
        nodes_per_layer=int(x[:, 1]),
        activation_fn=int(x[:, 2]),
        learning_rate=float(x[:, 3]))
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]


# #### Optimizer instance

# optimizer
opt_mnist = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

# #### Running optimization

# optimize mnist model
opt_mnist.run_optimization(max_iter=10)

# #### The output

# print optimized mnist model
print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
\t{6}:\t{7}
""".format(bounds[0]["name"], opt_mnist.x_opt[0],
           bounds[1]["name"], opt_mnist.x_opt[1],
           bounds[2]["name"], opt_mnist.x_opt[2],
           bounds[3]["name"], opt_mnist.x_opt[3]))
print("optimized loss: {0}".format(opt_mnist.fx_opt))