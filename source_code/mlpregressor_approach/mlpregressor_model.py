import pandas as pd
from sklearn.neural_network import MLPRegressor

def mlpregressor_model(X, y):
    model = MLPRegressor(hidden_layer_sizes=(10,),  # Number of neurons in each hidden layer
                         activation='relu',           # Activation function
                         solver='adam',               # Optimizer
                         max_iter=100,                # Maximum number of iterations
                         random_state=42)
    model.fit(X, y)
    return model
