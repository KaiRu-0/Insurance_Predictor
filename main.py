from google.colab import files
import data_processing as dp
import model as mdl
import prediction as prd

def main():
    # Uploading and reading the dataset
    uploaded = files.upload()
    
    # Assuming 'insurance.csv' is the name of the uploaded file
    filename = 'insurance.csv'
    data = dp.load_data(filename)
    
    # Display the first few rows and basic info
    print(data.head())
    print(data.shape)
    print(data.info())
    print(data.isnull().sum())
    
    # Display unique values in specific columns
    print(data['sex'].value_counts())
    print(data['smoker'].value_counts())
    print(data['region'].value_counts())
    
    # Preprocess data
    data = dp.preprocess_data(data)
    print(data.head())
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = dp.split_data(data)
    print(X_train.shape, X_test.shape)
    
    # Train the model
    model = mdl.train_model(X_train, Y_train)
    
    # Evaluate the model
    mdl.evaluate_model(model, X_train, Y_train, X_test, Y_test)
    
    # Making a single prediction
    input_data = (19, 1, 27.9, 0, 1, 0, 0)  # Example input
    prediction = prd.make_prediction(model, input_data)
    print('Predicted insurance charge:', prediction)

if __name__ == "__main__":
    main()
