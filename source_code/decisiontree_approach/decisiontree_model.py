import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def decisiontree_model(X, y):
    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model
