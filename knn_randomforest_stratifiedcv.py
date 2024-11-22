import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

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

# Train Random Forest Classifier with Stratified Cross-Validation
def train_classifier_with_stratified_cv(train_data, train_labels, cv_folds=5):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    stratified_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(classifier, train_data, train_labels, cv=stratified_cv, scoring='accuracy')
    print(f"Stratified Cross-Validation Accuracy Scores for {cv_folds} folds: {cv_scores}")
    print(f"Mean Stratified Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")
    
    classifier.fit(train_data, train_labels)
    return classifier

def predict_test_labels(classifier, test_data):
    predictions = classifier.predict(test_data)
    return predictions

def save_predictions(predictions, output_directory, output_filename):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_path = os.path.join(output_directory, output_filename)
    np.savetxt(output_file_path, predictions, fmt='%d')

def handle_classification_task(train_data_file, train_label_file, test_data_file, output_directory, output_filename):
    train_data, train_labels, test_data = load_data(train_data_file, train_label_file, test_data_file)
    imputed_train_data, imputed_test_data = impute_missing_values(train_data, test_data)
    classifier = train_classifier_with_stratified_cv(imputed_train_data, train_labels)
    predictions = predict_test_labels(classifier, imputed_test_data)
    save_predictions(predictions, output_directory, output_filename)

# Execution
handle_classification_task(
    train_data_file="training_data/TrainData1.txt",
    train_label_file="training_data/TrainLabel1.txt",
    test_data_file="testing_data/TestData1.txt",
    output_directory="output_files",
    output_filename="TestResult1.txt"
)

handle_classification_task(
    train_data_file="training_data/TrainData2.txt",
    train_label_file="training_data/TrainLabel2.txt",
    test_data_file="testing_data/TestData2.txt",
    output_directory="output_files",
    output_filename="TestResult2.txt"
)

handle_classification_task(
    train_data_file="training_data/TrainData3.txt",
    train_label_file="training_data/TrainLabel3.txt",
    test_data_file="testing_data/TestData3.txt",
    output_directory="output_files",
    output_filename="TestResult3.txt"
)

handle_classification_task(
    train_data_file="training_data/TrainData4.txt",
    train_label_file="training_data/TrainLabel4.txt",
    test_data_file="testing_data/TestData4.txt",
    output_directory="output_files",
    output_filename="TestResult4.txt"
)

handle_classification_task(
    train_data_file="training_data/TrainData5.txt",
    train_label_file="training_data/TrainLabel5.txt",
    test_data_file="testing_data/TestData5.txt",
    output_directory="output_files",
    output_filename="TestResult5.txt"
)
