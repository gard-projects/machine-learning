import pandas as pd
import numpy as np

class LinReg():
    def __init__(self, epoch=10, learning_rate=0.01):
        self.epoch = epoch
        self.learning_rate = learning_rate
        
    def fit(self, training_set):
        # Intialize weight matrix 
        # Weight matrix: (m+1, 1)
        # Feature matrix: (n, m+1)
        m = training_set.shape[1] - 1 # number of features (indepedent variables)
        n = training_set.shape[0] # number of samples (observations)
        
        weight_mtx = np.random.rand(m+1, 1)
        feature_mtx = np.hstack((np.ones((n,1)), training_set[:, :m])) # Add bias term
        actual_response = training_set[:, m:]
        
        # Calculate predicted response
        predicted_response = np.dot(feature_mtx, weight_mtx)
        self.loss_function(actual_response, predicted_response, m)
    
    def loss_function(self, actual_response, predicted_response, m: int) -> float:
        j = (1/(2*m)) * np.sum((predicted_response - actual_response)**2) # Mean Squared Error
        return j
    
    def gradient_descent(self):
        # Something here...
        return
    
if __name__ == "__main__":
    # Load dataset
    training_data = pd.read_csv("dataset/l1_train.csv").to_numpy()
    testing_data = pd.read_csv("dataset/l1_test.csv").to_numpy()
    
    # Linear regression
    linreg = LinReg(epoch=5, learning_rate=0.01)
    linreg.fit(training_data)
    
    
    
