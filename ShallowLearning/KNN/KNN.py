import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
import pandas as pd

# K-Nearest Neighbors (KNN) algorithm implementation
class KNN:
    
    
    def __init__(self, k: int):
        self.k = k
        self.model_accuracy = 0
        
    
    
    def fit(self, training_set: np.ndarray, testing_set: np.ndarray, 
            distance_metric: str='euclidean'):
        m = training_set.shape[1]-1 # Number of predictors
        
        self.X_features = training_set[:, :m]
        self.y = training_set[:, m]
        self.testing_set = testing_set
        self.actual_response = testing_set[:, m:]
        encoded_y = self.label_encoding(self.y)
        
        # Encode categorical features
        self.encode_features(None, m, encoded_y)
        self.encode_features(self.testing_set[:, :m], m, encoded_y, 'testing')

        # Implement KNN algorithm (compute distances and find k nearest neighbors)
        self.predict(self.testing_set, distance_metric)   
    
    
    def encode_features(self, features: np.ndarray , m: int, encoded_y: np.ndarray, set_type: str='training'):
       # Encoding variables
        indices_one_hot = [1, 5, 7, 8, 9]
        indices_target_encoding = [3, 6, 13]
        # Standardize numerical features
        numerical_features = [0, 2, 4, 10, 11, 12]

        if set_type == 'training':
            encoded_columns = np.zeros((self.X_features.shape[0],0))
            remaining_columns = np.zeros((self.X_features.shape[0],0))
           
           # Standardize numerical features ....
            
        
            for i in range(m):
                if i in indices_one_hot:
                    encoded_columns = np.concatenate((encoded_columns, self.one_hot_encoding(i)), axis=1)
                elif i in indices_target_encoding:
                    encoded_columns = np.concatenate((encoded_columns, self.target_encoding(i, encoded_y)), axis=1)
                else:
                    remaining_columns = np.concatenate((remaining_columns, self.X_features[:, i].reshape(-1,1)), axis=1)
            self.X_features = np.concatenate((encoded_columns, remaining_columns), axis=1)
        
        else:
            encoded_columns = np.zeros((features.shape[0],0))
            remaining_columns = np.zeros((features.shape[0],0))
        
            for i in range(m):
                if i in indices_one_hot:
                    encoded_columns = np.concatenate((encoded_columns, self.one_hot_encoding(i, features, 'testing')), axis=1)
                elif i in indices_target_encoding:
                    encoded_columns = np.concatenate((encoded_columns, self.target_encoding(i, encoded_y, features, 'testing')), axis=1)
                else:
                    remaining_columns = np.concatenate((remaining_columns, features[:, i].reshape(-1,1)), axis=1)
            self.testing_set = np.concatenate((encoded_columns, remaining_columns), axis=1)

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray, batch_size: int=1000):
        # Use batch processing to compute Euclidean distance, due to the large number of samples
        m, n = x1.shape[0], x2.shape[0]
        distances = np.empty((m, n))
      
        for i in range (0, m, batch_size):
            end_i_index = min(i + batch_size, m)
            x1_batch = x1[i:end_i_index, :]
            
            for j in range(0, n, batch_size):
                end_j_index = min(j + batch_size, n)
                x2_batch = x2[j:end_j_index, :]
                
                # Compute distances, result is a 2D numpy array
                dist_batch = np.sqrt(np.sum((x1_batch[np.newaxis, :, :].astype(float) - x2_batch[:, np.newaxis, :].astype(float))**2,axis=2))
                distances[i:end_i_index, j:end_j_index] = dist_batch.T
        return distances
    
    def manhattan_distance(self, x1: np.ndarray, x2: np.ndarray):
        d = np.sum(np.abs(x1-x2), axis=1)
        return d

        c = np.dot(x1, x2.T) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))
        return c
    
    def one_hot_encoding(self, column_index: int, features: np.ndarray=None, set_type: str='training'):
        if set_type == 'training':
            column = self.X_features[:, column_index]
        else: 
            column = features[:, column_index]
        
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
    
    def target_encoding(self, column_index: int, target: np.ndarray, features: np.ndarray=None, set_type: str='training'):
        if set_type == 'training':
            column = self.X_features[:, column_index]
        else:
            column = features[:, column_index]
        
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
    
    def predict(self, testing_set: np.ndarray, distance_metric: str):
        m = testing_set.shape[1]-1
        actual_response = testing_set[:, m:]
        distances = None
        predicted_response = []
        
        if distance_metric == 'euclidean':
            distances = self.euclidean_distance(testing_set, self.X_features)
            
        # Find the k nearest neighbors
        neighbors = np.argsort(distances, axis=1)[:, :self.k]
        neighbor_labels = self.y[neighbors]
        for i in range(testing_set.shape[0]):
            unique_labels, counts = np.unique(neighbor_labels[i,:], return_counts=True)
            majority_vote = unique_labels[np.argmax(counts)]
            predicted_response.append(majority_vote)
        
        self.model_accuracy = self.accuracy(self.actual_response, np.array(predicted_response))
    
    def accuracy(self, actual: np.ndarray, predicted: np.ndarray):
        return np.mean(actual == predicted)
    
     
if __name__ == '__main__':
    custom_model = KNN(k=5)
   
    data = pd.read_csv('dataset/adult.csv')
    X = data.drop('income', axis=1)
    y = data['income']
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Sklearn model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=10)
    # Indices of features to encode
    numerical_features = [0, 2, 4, 10, 11, 12]
    categorical_features = [1, 5, 7, 8, 9]
    target_encoding_features = [3, 6, 13]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('one-hot', OneHotEncoder(), categorical_features),
            ('target', TargetEncoder(), target_encoding_features),
            ('standard', StandardScaler(), numerical_features)
        ]
    )
    
    # Model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])
    
    pipeline.fit(X_train, y_train)
    sklearn_accuracy = pipeline.score(X_test, y_test)
    
    
    # Custom model
    data = data.to_numpy()
    split_index = int(0.8*data.shape[0])
    training_set, testing_set = data[:split_index], data[split_index:]
    custom_model.fit(training_set, testing_set)
    print(f"Accuracy of custom model: {custom_model.model_accuracy}")
    print(f"Accuracy of sklearn model: {sklearn_accuracy}")