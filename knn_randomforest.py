import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

# Function to load the dataset from .txt files
def load_data(train_data_file, train_label_file, test_data_file):
    train_data = pd.read_csv(train_data_file, sep=r'[,\s;]+', header=None, engine='python')
    train_labels = pd.read_csv(train_label_file, sep=r'[,\s;]+', header=None, engine='python').values.ravel()
    test_data = pd.read_csv(test_data_file, sep=r'[,\s;]+', header=None, engine='python')
    
    return train_data, train_labels, test_data

# Imputation function
def impute_missing_values(train_data, test_data):
    train_data.replace(1.00000000000000e+99, np.nan, inplace=True)
    test_data.replace(1.00000000000000e+99, np.nan, inplace=True)
    
    imputer = KNNImputer(n_neighbors=5)
    imputed_train_data = imputer.fit_transform(train_data)
    imputed_test_data = imputer.transform(test_data)
    
    return imputed_train_data, imputed_test_data

# Train classifier using Naive Bayes
def train_classifier(train_data, train_labels):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(train_data, train_labels)
    return classifier

# Predict test labels
def predict_test_labels(classifier, test_data):
    predictions = classifier.predict(test_data)
    return predictions

# Save predictions function (with directory path option)
def save_predictions(predictions, output_directory, output_filename):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_path = os.path.join(output_directory, output_filename)
    np.savetxt(output_file_path, predictions, fmt='%d')

# Main function to handle the complete task for one dataset
def handle_classification_task(train_data_file, train_label_file, test_data_file, output_directory, output_filename):
    # Step 1: Load the data
    train_data, train_labels, test_data = load_data(train_data_file, train_label_file, test_data_file)
    
    # Step 2: Impute missing values
    imputed_train_data, imputed_test_data = impute_missing_values(train_data, test_data)
    
    # Step 3: Train classifier
    classifier = train_classifier(imputed_train_data, train_labels)
    
    # Step 4: Predict test data labels
    predictions = predict_test_labels(classifier, imputed_test_data)
    
    # Step 5: Save the predictions
    save_predictions(predictions, output_directory, output_filename)

# Loop through datasets 1 to 6
for i in range(1, 7):
    handle_classification_task(
        train_data_file=f"training_data/TrainData{i}.txt",
        train_label_file=f"training_data/TrainLabel{i}.txt",
        test_data_file=f"testing_data/TestData{i}.txt",
        output_directory="output_files/rf_predictions",
        output_filename=f"DeninaChungBangResult{i}.txt"
    )
