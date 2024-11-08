import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

# Loop through dataset numbers from 1 to 6
for i in range(1, 7):
    # Generate the filenames dynamically based on the loop index
    train_data_file = f"training_data/TrainData{i}.txt"
    train_labels_file = f"training_data/TrainLabel{i}.txt"
    test_data_file = f"testing_data/TestData{i}.txt"
    output_file = f"DeninaClassification{i}.txt"

    # Load data with whitespace delimiter
    train_data = pd.read_csv(train_data_file, sep='\s+|,|\t', header=None, engine='python')
    train_labels = pd.read_csv(train_labels_file, sep='\s+|,|\t', header=None, engine='python').values.ravel()
    test_data = pd.read_csv(test_data_file, sep='\s+|,|\t', header=None, engine='python')

    # Clean data by stripping extra whitespace
    train_data = train_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    test_data = test_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Convert missing values
    missing_value = 1.00000000000000e+99
    train_data.replace(missing_value, np.nan, inplace=True)
    test_data.replace(missing_value, np.nan, inplace=True)

    # Convert to numpy arrays to avoid feature name issues
    train_data_np = train_data.to_numpy()
    test_data_np = test_data.to_numpy()

    # Impute missing values
    knn_imputer = KNNImputer(n_neighbors=5)
    train_data_imputed = knn_imputer.fit_transform(train_data_np)
    test_data_imputed = knn_imputer.transform(test_data_np)

    # Train Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_data_imputed, train_labels)

    # Predict and Save Results
    predictions = rf_classifier.predict(test_data_imputed)
    pd.DataFrame(predictions, columns=['PredictedLabel']).to_csv(output_file, index=False, header=False)

    print(f"Finished processing dataset {i}")
