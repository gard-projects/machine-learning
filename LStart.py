import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinReg():
    '''
    A class used to present a linear regression model
    
    Attributes
    ----------
    epochs : int
        Number of iterations for the gradient descent
    learning_rate : int
        The step size for the gradient descent
    
    Methods
    -------
    fit(X_features, y)
        Estimates the weights for the linear regression model
    gradient(X, y, n)
        Calculates the gradient of the cost function given the predictors and response
    cost_function(X, y)
        Calculates the distance score using the MSE cost function
    predict(X)
        Predicts the estimated response for the given predictors
    standardize(X)
        Standardizes the a given matrix X
    '''
    def __init__(self, epochs: int, learning_rate: int):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.theta= None
        self.cost_history = [] # Keep track of the distance score
        
    def fit(self, X_features: np.ndarray, y: np.ndarray):
        '''Find the weights for the linear regression model'''
        n = X_features.shape[0] # Number of samples
        m = X_features.shape[1] # Number of predictors (features)
        
        
        # Setup design matrix
        X_features = self.standardize(X_features) # Standardize the predictors
        X = np.c_[np.ones(n), X_features]
        
        # Random initialization of weights
        self.theta = np.random.rand(m+1, 1)
        
    
        # Gradient Descent
        for _ in range(self.epochs):
            step =  self.learning_rate * self.gradient(X, y, n)
            self.theta = self.theta - step
            self.cost_history.append(self.cost_function(X, y))
    
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
        e = y_est - y 
        J = 1/(2*n) * np.dot(e.T, e).item() # Translate 1x1 matrix into a scalar
        return J 
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predicts the estimated response for the given input'''
        y_est = np.dot(X, self.theta)
        return y_est
    
    def r_square(self, X: np.ndarray, y: np.ndarray) -> float:
        '''Calculates the R^2 score for the given input and response'''
        X = self.standardize(X)
        y_est = self.predict(np.c_[np.ones(X.shape[0]), X])
        e = y - y_est
        # Residual sum of squares (SS_res)
        SS_res = np.dot(e.T, e).item() 
        
        y_bar = np.mean(y)
        s = y - y_bar
        # Total sum of squares (SS_tot)
        SS_tot = np.dot(s.T, s).item()
        
        r_2 = 1 - SS_res/SS_tot
        return r_2
        
    
    def standardize(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std
        return X


        
if __name__ == "__main__":
    # Handle csv file
    training_set = pd.read_csv('./dataset/l1_train.csv').to_numpy()
    testing_set = pd.read_csv('./dataset/l1_test.csv').to_numpy()
    m = training_set.shape[1]-1 # Number of predictors (features)
    
    # Model training and results
    model = LinReg(epochs=10000, learning_rate=0.001)
    model.fit(X_features=training_set[:,:m], y=training_set[:,m:])
    
    # Compare with sklearn implementation
    reg = LinearRegression().fit(training_set[:,:m], training_set[:,m:])
    r_2_sklearn = reg.score(testing_set[:,:m], testing_set[:,m:])
    print(r_2_sklearn)
    
    # Image of the cost history
    plt.figure()
    plt.plot(np.arange(1, model.epochs + 1), model.cost_history, 'r', label='Cost history')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid()
    plt.savefig('cost_history.png')
    
    
    predicted_response = np.dot(model.standardize(testing_set[:,:m]), model.theta[1:]) + model.theta[0]
    
    # Image of prediction
    plt.figure()
    plt.scatter(testing_set[:,:m], testing_set[:,m:], label='True response', color='blue')
    plt.plot(testing_set[:,:m], predicted_response, label='Predicted response', color='red')
    plt.title('Linear Regression')
    plt.xlabel('Predictor (X)')
    plt.ylabel('Response (y)')
    plt.legend()
    plt.savefig('prediction.png')

    model_r2 = model.r_square(testing_set[:,:m], testing_set[:,m:])
    print(model_r2)
    

