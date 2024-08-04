import pandas as pd
from sklearn.linear_model import LinearRegression

def baseline_model(X_processed, y):
    reg = LinearRegression().fit(X_processed, y)
    return reg
