import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Allowing virtual environment to be used: Set-ExecutionPolicy Unrestricted -Scope Process

class LinReg():
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.theta= np.zeros(0)
        self.cost_history = [] # Keep track of the distance score
        
    def fit(self, X_features: np.ndarray, y: np.ndarray):
        '''Find the weights for the linear regression model'''
        n = X_features.shape[0] # Number of samples
        m = X_features.shape[1] # Number of predictors (features)
        
        # Setup design matrix
        X = np.c_[np.ones(n), X_features]
        
        # Random initialization of weights
        self.theta = np.random.rand(m+1, 1)
    
        # Gradient Descent
        for _ in range(self.epochs):
            self.theta -= self.learning_rate * self.gradient(X, y, n)
            print("Grad:", self.gradient(X, y, n))
            self.cost_history.append(self.cost_function(X, y, n))
        return 0
    
    def gradient (self, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        '''Calculates the gradient for the given matrices X and y'''
        y_est = self.predict(X)
        e = y_est - y # Error vector
        g = (1/n) * np.dot(X.T, e)
        return g
    
    def cost_function(self, X: np.ndarray, y: np.ndarray, n: int) -> float:
        '''Calculates the distance score using the MSE cost function''' 
        y_est = self.predict(X)
        e = y_est - y # Error vector
        J = (1/(2*n)) * np.dot(e.T, e)
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
    model = LinReg(epochs=4, learning_rate=0.001)
    model.fit(X_features=training_set[:,:m], y=training_set[:,m:])
    print("Model weights: ", model.theta)
    print("Cost history: ", model.cost_history)
    
    
    # Compare with sklearn implementation
    reg = linear_model.LinearRegression().fit(training_set[:,:m], training_set[:,m:])
    print("Sklearn weights: ", reg.coef_, reg.intercept_)
    
    # Visualize the estimated model
    plt.scatter(training_set[:,:m], training_set[:,m:], color="blue", label="Training set")
    plt.plot(training_set[:,:m], model.predict(np.c_[np.ones(training_set.shape[0]), training_set[:,:m]]), color="red", label="Model")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()
