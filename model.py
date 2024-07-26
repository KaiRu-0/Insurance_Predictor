from sklearn.linear_model import LinearRegression
from sklearn import metrics

def train_model(X_train, Y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    """
    Evaluate the trained model using R squared metric.
    """
    # Prediction on training data
    training_data_prediction = model.predict(X_train)
    r2_train = metrics.r2_score(Y_train, training_data_prediction)
    print('R squared value (training data):', r2_train)

    # Prediction on testing data
    test_data_prediction = model.predict(X_test)
    r2_test = metrics.r2_score(Y_test, test_data_prediction)
    print('R squared value (testing data):', r2_test)
