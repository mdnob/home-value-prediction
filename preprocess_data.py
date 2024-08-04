import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df_train):
    # Select numeric and categorical columns
    numeric_features = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features.remove('logerror')  # Remove target column
    categorical_features = df_train.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Convert boolean columns to strings
    for col in categorical_features:
        df_train[col] = df_train[col].astype(str)

    # Split the data into features and target
    X = df_train.drop(columns=['logerror'])
    y = df_train['logerror']

    # Preprocessing for numeric data: impute missing values and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    # Preprocessing for categorical data: impute missing values and encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor