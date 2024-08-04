import pandas as pd

def load_and_clean_data(properties_2016_path, train_2016_path, properties_2017_path, train_2017_path):
    # Function to load data and drop columns with more than 80% null values
    def load_and_drop_nulls(properties_path, train_path):
        properties = pd.read_csv(properties_path, low_memory=False)
        train = pd.read_csv(train_path, low_memory=False)

        # Drop columns with more than 80% null values
        threshold = 0.8 * len(properties)
        properties = properties.dropna(axis=1, thresh=threshold)

        # Merge datasets
        df_train = train.merge(properties, how='left', on='parcelid')
        return df_train

    # Load and clean 2016 data
    df_train_2016 = load_and_drop_nulls(properties_2016_path, train_2016_path)
    
    # Load and clean 2017 data
    df_train_2017 = load_and_drop_nulls(properties_2017_path, train_2017_path)

    # Combine 2016 and 2017 data
    df_combined = pd.concat([df_train_2016, df_train_2017], ignore_index=True)

    return df_combined
