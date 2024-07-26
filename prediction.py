import numpy as np

def make_prediction(model, input_data):
    """
    Make a prediction for a single data point.
    """
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]
