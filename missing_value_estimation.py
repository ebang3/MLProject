import numpy as np
from sklearn.impute import KNNImputer

def load_data(file_path):
    data = np.loadtxt(file_path)
    # Replace missing values (1.00000000000000e+99) with NaN for processing
    data[data == 1.00000000000000e+99] = np.nan
    return data

def impute_missing_values(data):
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(data)
    return imputed_data

def save_data(file_path, data):
    np.savetxt(file_path, data, fmt='%f')

# Dataset 1
data1 = load_data('missing_data\MissingData1.txt')
imputed_data1 = impute_missing_values(data1)
save_data('output_files\Dataset1_Completed.txt', imputed_data1)

# Dataset 2
data2 = load_data('missing_data\MissingData2.txt')
imputed_data2 = impute_missing_values(data2)
save_data('output_files\Dataset2_Completed.txt', imputed_data2)
