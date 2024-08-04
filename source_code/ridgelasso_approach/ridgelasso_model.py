import pandas as pd
from sklearn.linear_model import Ridge, Lasso

def ridge_model(X, y):
    ridge_model = Ridge(alpha=1.0)  # The alpha can be adjusted
    ridge_model.fit(X, y)
    return ridge_model
    
def lasso_model(X, y):
    lasso_model = Lasso(alpha=0.1)  # The alpha can be adjusted
    lasso_model.fit(X, y)
    return lasso_model
