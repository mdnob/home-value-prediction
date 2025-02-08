import pandas as pd
from sklearn.linear_model import Ridge

def ridge_model(X, y):
    ridge_model = Ridge(alpha=1.0)  # The alpha can be adjusted
    ridge_model.fit(X, y)
    return ridge_model
