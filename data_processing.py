import pandas as pd

def load_data(filename):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(filename)

def preprocess_data(data):
    """
    Preprocess the data by encoding categorical features and handling missing values.
    """
    # Encoding 'sex' and 'smoker' columns
    data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    data.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)

    # One-hot encoding 'region' column
    data = pd.get_dummies(data, columns=['region'], drop_first=True)
    
    return data

def split_data(data):
    """
    Split the data into features and target, then into training and testing sets.
    """
    X = data.drop(columns='charges', axis=1)
    Y = data['charges']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    return X_train, X_test, Y_train, Y_test
