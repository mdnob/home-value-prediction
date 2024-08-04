import pandas as pd
from sklearn.linear_model import LinearRegression

def baseline_model(X, y):
    reg = LinearRegression().fit(X, y)
    return reg
