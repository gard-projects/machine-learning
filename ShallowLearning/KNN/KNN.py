import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
import pandas as pd


class KNN:
    '''
    A class used to present a K-Nearest Neighbors model
    
    Attributes
    ----------
    k : int
        Number of neighbors to consider
    model_accuracy : float
        Accuracy of the model on the testing set
    X_features : np.ndarray
        A matrix with n samples and m predictors, shape (n, m) for training set
    y : np.ndarray
        A vector with n responses, shape (n, 1) for training set
    testing_set : np.ndarray
        A matrix with n samples and m predictors, shape (n, m) for testing set
    mean : np.ndarray
        Mean of the numerical features
    std : np.ndarray
        Standard deviation of the numerical features
    
    Methods
    -------
    fit(training_set, testing_set, distance_metric='euclidean')
        Fits the KNN model, performs scaling, encoding, and computes the accuracy
    encode_features(features, m, encoded_y, set_type='training')
        Encodes the categorical features for both training and testing sets
    euclidean_distance(x1, x2, batch_size=1000)
        Computes the Euclidean distance between two matrices, using batch processing
    manhattan_distance(x1, x2, batch_size=1000)
        Computes the Manhattan distance between two matrices, using batch processing
    one_hot_encoding(column_index, features=None, set_type='training')
        Encodes the categorical features using one-hot encoding
    label_encoding(column)
        Encodes the response variable using label encoding
    target_encoding(column_index, target, features=None, set_type='training')
        Encodes the categorical features using target encoding
    predict(testing_set, distance_metric)
        Predicts the labels for the samples in the testing set
    accuracy(actual, predicted)
        Computes the accuracy of the model 
    '''
    
    def __init__(self, k: int):
        """
        Parameters
        ----------
        k : int
            Number of neighbors to consider
        """
        self.k = k
        self.model_accuracy = 0
        
    
    def fit(self, training_set: np.ndarray, testing_set: np.ndarray, distance_metric: str='euclidean'):
        '''
        The main method to fit the KNN model, perform scaling, encoding, and compute the accuracy
        
        Parameters
        ----------
        training_set: np.ndarray
            A matrix with n samples and m predictors, shape (n, m) for training set
        testing_set: np.ndarray
            A matrix with n samples and m predictors, shape (n, m) for testing set
        distance_metric: str
            The distance metric to use, either 'euclidean' or 'manhattan'
        '''
        
        m = training_set.shape[1] - 1  # Number of predictors
        
        self.X_features = training_set[:, :m]
        self.y = training_set[:, m]
        self.testing_set = testing_set
        self.actual_response = testing_set[:, m]
        
        encoded_y = self.label_encoding(self.y)
        
        numerical_features = [0, 2, 4, 10, 11, 12]
        self.mean = np.mean(self.X_features[:, numerical_features].astype(float), axis=0)
        self.std = np.std(self.X_features[:, numerical_features].astype(float), axis=0)
        
        self.X_features[:, numerical_features] = (self.X_features[:, numerical_features] - self.mean) / self.std
        self.testing_set[:, numerical_features] = (self.testing_set[:, numerical_features] - self.mean) / self.std
        
        self.encode_features(self.X_features, m, encoded_y)
        self.encode_features(self.testing_set[:, :m], m, encoded_y, 'testing')
        
        self.predict(self.testing_set, distance_metric) 
    
    
    def encode_features(self, features: np.ndarray , m: int, encoded_y: np.ndarray, set_type: str='training'):
        '''
        Encodes the categorical features for both training and testing sets
        
        Parameters
        ----------
        features: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the categorical variables of the testing set
        m: int
            Number of predictors
        encoded_y: np.ndarray
            Encoded response variable
        set_type: str
            The type of the set, either 'training' or 'testing'   
        '''
        indices_one_hot = [1, 5, 7, 8, 9]
        indices_target_encoding = [3, 6, 13]

        # Training set
        if set_type == 'training':
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
        
        else:
            # Testing set
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

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray, batch_size: int=1000) -> np.ndarray:
        '''
        Computes the Euclidean distance between two matrices, using batch processing
        
        Parameters
        ----------
        x1: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the testing set
        x2: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the training set
        batch_size: int
            The size of the batch to use for processing
            
        Returns
        -------
        distances: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the Euclidean distances
        '''
        
        m, n = x1.shape[0], x2.shape[0]
        distances = np.empty((m, n))
      
        for i in range (0, m, batch_size):
            end_i_index = min(i + batch_size, m)
            x1_batch = x1[i:end_i_index, :]
            
            for j in range(0, n, batch_size):
                end_j_index = min(j + batch_size, n)
                x2_batch = x2[j:end_j_index, :]
                
                dist_batch = np.sqrt(np.sum((x1_batch[np.newaxis, :, :].astype(float) - x2_batch[:, np.newaxis, :].astype(float))**2,axis=2))
                distances[i:end_i_index, j:end_j_index] = dist_batch.T
        
        return distances
    
    def manhattan_distance(self, x1: np.ndarray, x2: np.ndarray, batch_size: int=1000) -> np.ndarray:
        '''
        Computes the Manhattan distance between two matrices, using batch processing
        
        Parameters
        ----------
        x1: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the testing set
        x2: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the training set
        batch_size: int
            The size of the batch to use for processing
            
        Returns
        -------
        distances: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the Manhattan distances
        
        '''
        m, n = x1.shape[0], x2.shape[0]
        distances = np.empty((m, n))

        for i in range(0, m, batch_size):
            end_i_index = min(i + batch_size, m)
            x1_batch = x1[i:end_i_index, :]

            for j in range(0, n, batch_size):
                end_j_index = min(j + batch_size, n)
                x2_batch = x2[j:end_j_index, :]

                # Compute Manhattan distances
                dist_batch = np.sum(np.abs(x1_batch[:, np.newaxis, :].astype(float) - x2_batch[np.newaxis, :, :].astype(float)), axis=2)
                distances[i:end_i_index, j:end_j_index] = dist_batch

        return distances


        c = np.dot(x1, x2.T) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))
        return c
    
    def one_hot_encoding(self, column_index: int, features: np.ndarray=None, set_type: str='training'):
        '''
        Encodes the categorical features using one-hot encoding
        
        Parameters
        ----------
        column_index: int
            The index of the column to encode
        features: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the testing set (default is None)
        set_type: str
            The type of the set, either 'training' or 'testing' (default is 'training')
            
        Returns
        -------
        encoded_column: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the encoded column
        '''
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
        '''
        Encodes the response variable using label encoding
        
        Parameters
        ----------
        column: np.ndarray
            A vector with n responses, shape (n, 1)
        
        Returns
        -------
        result: np.ndarray
            A vector with n responses, shape (n, 1), represents the encoded response variable
        '''
        u = np.unique(column)
        v = {value: i for i, value in enumerate(u)}
        result = np.array([v[value] for value in column])
        return result
    
    def target_encoding(self, column_index: int, target: np.ndarray, features: np.ndarray=None, set_type: str='training'):
        '''
        Encodes the categorical features using target encoding
        
        Parameters
        ----------
        column_index: int
            The index of the column to encode
        target: np.ndarray
            A vector with n responses, shape (n, 1)
        features: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the testing set (default is None)
        set_type: str
            The type of the set, either 'training' or 'testing' (default is 'training')
        
        Returns
        -------
        encoded_column: np.ndarray
            A matrix with n samples and m predictors, shape (n, m), represents the encoded column
        '''
        if set_type == 'training':
            column = self.X_features[:, column_index]
        else:
            column = features[:, column_index]
        
        unique_values = np.unique(column)
        encoded_column = np.zeros((column.shape[0], 1))
        for u in unique_values:
                encoded_column[np.where(column == u)] = np.mean(target[np.where(column == u)])
        
        return encoded_column
     
    def predict(self, testing_set: np.ndarray, distance_metric: str):
        '''
        Finds the K nearest neighbors and predicts the labels for the samples in the testing set
        
        Parameters
        ----------
        testing_set: np.ndarray
            A matrix with n samples and m predictors, shape (n, m) for testing set
        distance_metric: str
            The distance metric to use, either 'euclidean' or 'manhattan'
        '''
        m = testing_set.shape[1]-1
        actual_response = testing_set[:, m:]
        distances = None
        predicted_response = []
        
        if distance_metric == 'euclidean':
            distances = self.euclidean_distance(testing_set, self.X_features)
        elif distance_metric == 'manhattan':
            distances = self.manhattan_distance(testing_set, self.X_features)
            
        # Find the k nearest neighbors
        neighbors = np.argsort(distances, axis=1)[:, :self.k]
        neighbor_labels = self.y[neighbors]
        for i in range(testing_set.shape[0]):
            unique_labels, counts = np.unique(neighbor_labels[i,:], return_counts=True)
            majority_vote = unique_labels[np.argmax(counts)]
            predicted_response.append(majority_vote)
        
        self.model_accuracy = self.accuracy(self.actual_response, np.array(predicted_response))
    
    def accuracy(self, actual: np.ndarray, predicted: np.ndarray):
        '''
        Computes the accuracy of the model
        
        Parameters
        ----------
        actual: np.ndarray
            A vector with n responses, shape (n, 1)
        predicted: np.ndarray
            A vector with n responses, shape (n, 1)
        
        Returns
        -------
        mean: float
            The estimated accuracy of the model
        '''
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
    
    # Pipeline for data preprocessing and model training
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