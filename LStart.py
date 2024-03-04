import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinReg():
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.theta= None
        self.cost_history = [] # Keep track of the distance score
        self.predicted = None
        
    def fit(self, X_features: np.ndarray, y: np.ndarray, x_test=None):
        '''Find the weights for the linear regression model'''
        n = X_features.shape[0] # Number of samples
        m = X_features.shape[1] # Number of predictors (features)
        
        # Normalize data
        mean = np.mean(X_features, axis=0)
        std = np.std(X_features, axis=0)
        X_features = (X_features - mean) / std
        x_test = (x_test - mean) / std
        
        # Setup design matrix
        X = np.c_[np.ones(n), X_features]
        Y_est = np.c_[np.ones(x_test.shape[0]), x_test]
        
        # Random initialization of weights
        self.theta = np.random.rand(m+1, 1)
        
    
        # Gradient Descent
        for _ in range(self.epochs):
            step =  self.learning_rate * self.gradient(X, y, n)
            self.theta = self.theta - step
            self.cost_history.append(self.cost_function(X, y))
            
        self.predicted = self.predict(Y_est)
        return 0
    
    def gradient (self, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        '''Calculates the gradient for the given matrices X and y'''
        y_est = self.predict(X)
        e = y_est - y # Error vector
        g = (1/n) * np.dot(X.T, e)
        return g
    
    def cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        '''Calculates the distance score using the MSE cost function''' 
        n = len(y) 
        y_est = self.predict(X)
        e = y_est - y # Error vector
        J = 1/(2*n) * np.dot(e.T, e).flatten()
        return J
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predicts the estimated response for the given input'''
        y_est = np.dot(X, self.theta)
        return y_est
        
        
if __name__ == "__main__":
    # Handle csv file
    training_set = pd.read_csv('./dataset/l1_train.csv').to_numpy()
    testing_set = pd.read_csv('./dataset/l1_test.csv').to_numpy()
    m = training_set.shape[1]-1 # Number of predictors (features)
    
    # Model training and results
    model = LinReg(epochs=10000, learning_rate=0.001)
    model.fit(X_features=training_set[:,:m], y=training_set[:,m:], x_test=testing_set[:,m:])
    
    # Compare with sklearn implementation
    reg = LinearRegression().fit(training_set[:,:m], training_set[:,m:])
    
    # Image of the cost history
    plt.figure()
    plt.plot(np.arange(1, model.epochs + 1), model.cost_history, 'r', label='Cost history')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig('cost_history.png')
    
    # Image of prediction
    plt.figure()
    plt.scatter(testing_set[:,:m], testing_set[:,m:], label='True response', color='blue')
    plt.plot(testing_set[:,:m], model.predicted, label='Predicted response', color='red')
    plt.title('Linear Regression')
    plt.xlabel('Predictor (X)')
    plt.ylabel('Response (y)')
    plt.legend()
    plt.savefig('prediction.png')

