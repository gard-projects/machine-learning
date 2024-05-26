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
        encoded_y = self.label_encoding(self.y)
        
        # Encoding variables
        indices_one_hot = [1, 5, 7, 8, 9]
        indices_target_encoding = [3, 6, 13]
        encoded_columns = np.zeros((self.X_features.shape[0],0))
        remaining_columns = np.zeros((self.X_features.shape[0],0))

        for i in range(m):
            if i in indices_one_hot:
                encoded_columns = np.concatenate((encoded_columns, self.one_hot_encoding(i)), axis=1)
            elif i in indices_target_encoding:
                encoded_columns = np.concatenate((encoded_columns, self.target_encoding(i, encoded_y)), axis=1)
            else:
                remaining_columns = np.concatenate((remaining_columns, self.X_features[:, i].reshape(-1,1)), axis=1)
        self.X_features = np.concatenate((encoded_columns, remaining_columns), axis=1)
        
        # Implement KNN algorithm (compute distances and find k nearest neighbors)

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray):
        d = np.sqrt(np.sum((x1-x2)**2, axis=1))
        return d
    
    def manhattan_distance(self, x1: np.ndarray, x2: np.ndarray):
        d = np.sum(np.abs(x1-x2), axis=1)
        return d
    
    def minkowski_distance(self, x1: np.ndarray, x2: np.ndarray, p: int):
        d = np.sum(np.abs(x1-x2)**p,axis=1)**(1/p)
        return d
    
    def cosine_similarity(self, x1: np.ndarray, x2: np.ndarray):
        c = np.dot(x1, x2.T) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))
        return c
    
    def one_hot_encoding(self, column_index: int):
        column = self.X_features[:, column_index]
        
        u = np.unique(column)
        encoded_column = np.zeros((column.shape[0], len(u)))
        for i, unique_value in enumerate(u):
            encoded_column[np.where(column == unique_value), i] = 1
        
        return encoded_column
           
    def label_encoding(self, column: np.ndarray) -> np.ndarray:
        u = np.unique(column)
        v = {value: i for i, value in enumerate(u)}
        result = np.array([v[value] for value in column])
        return result
    
    def target_encoding(self, column_index: int, target: np.ndarray):
        column = self.X_features[:, column_index]
        
        unique_values = np.unique(column)
        encoded_column = np.zeros((column.shape[0], 1))
        for u in unique_values:
                encoded_column[np.where(column == u)] = np.mean(target[np.where(column == u)])
        
        return encoded_column
    
    def k_folds(self, data: np.ndarray):
        # Shuffle samples in dataset
        np.random.shuffle(data)
        
        # Split data into k folds
        fold_size = data.shape[0] // self.k
        # An array of k folds
        folds = [data[i*fold_size:(i+1)*fold_size] for i in range(self.k)]
        print(data[self.k*fold_size:])
        print(data[self.k*fold_size:].shape)
        
        # Check if data can be split in equal folds
        if data.shape[0] % self.k != 0:
            folds[-1] = np.concatenate((folds[-1], data[self.k*fold_size:]))
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
    
    def leave_one_out_target_encoding(self, column: np.ndarray, target: np.ndarray):
        unique_categories = np.unique(column)
        
        # Map each category to respective indices in target variable
        category_indices = {c: np.where(column==c) for c in unique_categories}
        column_encoded = np.zeros_like(target)
        
        for category, indices in category_indices.items():
            category_sum = np.sum(target[indices])
            category_count = len(indices)-1
            
            if category_count == 0:
                # If there is only one sample in the category, use the global mean
                column_encoded[indices] = np.mean(target)
            else:
                # More than one sample in the category
                for i in indices:
                    column_encoded[i] = (category_sum - target[i]) / category_count
            
        return column_encoded
    
    
    
     
if __name__ == '__main__':
    custom_model = KNN(k=3)
    # Target income - dependent variable
    
    # Fetch data from csv
    data = pd.read_csv('dataset/adult.csv').to_numpy()
    split_index = int(0.8*data.shape[0])
    training_set, testing_set = data[:split_index], data[split_index:]
    custom_model.fit(training_set)
