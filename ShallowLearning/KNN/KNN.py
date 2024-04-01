import numpy as np
import pandas as pd

# K-Nearest Neighbors (KNN) algorithm implementation
class KNN:
    def __init__(self, k: int):
        self.k = k
        
    def fit(self, data: np.ndarray):
        m = data.shape[1]-1 # Number of predictors
        
        self.X_features = data[:, :m]
        self.y = data[:, m]
        
        self.k_fold_target_encoding(self.X_features, self.y)
        return 0

    def eculedian_distance(self):
        return 0
    
    def predict(self):
        return 0
    
    def score(self):
        return 0
    
    def one_hot_encoding(self, column: np.ndarray):
        u = np.unique(column)
        result = np.zeros((column.shape[0], len(u)))
        for i, unique_value in enumerate(u):
            result[np.where(column == unique_value), i] = 1
        return result
    
    def label_encoding(self, column: np.ndarray):
        u = np.unique(column)
        v = {value: i for i, value in enumerate(u)}
        result = np.array(v[value] for value in column)
        return result
    
    def target_encoding(self, column: np.ndarray, target: np.ndarray):
        unique_values = np.unique(column)
        result = np.zeros((column.shape[0], 1))
        for u in unique_values:
            result[np.where(column == u)] = np.mean(target[np.where(column == u)])
        return result
    
    def k_folds(self, data: np.ndarray):
        # Shuffle samples in dataset
        np.random.shuffle(data)
        
        # Split data into k folds
        fold_size = data.shape[0] // self.k
        # An array of k folds
        folds = [data[i*fold_size:(i+1)*fold_size] for i in range(k)]
        print(data[k*fold_size:])
        print(data[k*fold_size:].shape)
        
        # Check if data can be split in equal folds
        if data.shape[0] % k != 0:
            folds[-1] = np.concatenate((folds[-1], data[k*fold_size:]))
        return folds
    
    def k_fold_target_encoding(self, column: np.ndarray, target: np.ndarray):
        column_folds = self.k_folds(column, self.k)
        target_folds = self.k_folds(target, self.k)
        result = np.zeros(column.shape)
        
        for i in range(self.k):
            # Step 1: Split data into training and validation sets
            training_data = np.concatenate(column_folds[j] for j in range(self.k) if j != i)
            training_target = np.concatenate(target_folds[j] for j in range(self.k) if j != i)
            validation_data = column_folds[i]
            
            # Step 2: Find the unique categories in the training data
            unique_values = np.unique(training_data)
            
            # Step 3: Calculate the mean of the target variable for each category
            for u in unique_values:
                mean_target = np.mean(training_target[training_data == u]) # Boolean mask
                result[validation_data == u] = mean_target
        return result
    
    def leave_one_out_target_encoding(self):
        return 0
    
if __name__ == '__main__':
    custom_model = KNN(k=3)
    # Target income - dependent variable
    
    # Fetch data from csv
    training_data = pd.read_csv('dataset/adult.csv').to_numpy()
    custom_model.fit(training_data)
    
    # One hot encoding (create a column for each category)
    
    # Label encoding
    
    # Target encoding / Bayesian Mean Encoding
    
    # K-Fold Target Encoding (varient of Target Encoding)
    
    # Leave-One-Out Target Encoding (varient of Target Encoding)
