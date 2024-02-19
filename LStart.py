import pandas as pd
import numpy as np

# Allowing virtual environment to be used: Set-ExecutionPolicy Unrestricted -Scope Process

class LinReg():
    def __init__(self, epoch=10, learning_rate=0.001, stop_criterion=1e-6):
        self.theta= None # Weights matrix
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.stop_criterion = stop_criterion
        
    def fit(self, training_set):
      m = training_set.shape[1]-1   # number of predictors
      n = training_set.shape[0]     # number of samples
      previous_cost = 0
      
      # Split dataset into predictors and response
      X = training_set[0:n, 0:m]
      y = training_set[0:n, -1:]
      
      # Design matrix (include bias term)
      X = np.c_[(np.ones(n), X)]

      # Random initialization of weights
      self.theta = np.random.rand(m+1, 1)
         
      # Gradient descent
      for i in range(self.epoch):
          grad = self.gradient_descent(n, X, y)
          self.theta = self.theta - self.learning_rate * grad
          # Calculate the distance score (loss) for each epoch
          error = self.cost_function(n, X, y)
          print("Error estimate:", error, "Epoch:", i+1)
      
      return 0
  
    def cost_function(self, n, X, y):
        '''J(theta), uses mean squared error (MSE)'''
        # Altenrative: J = 1/(2*n) * np.sum(np.square(np.matmul(X,self.theta) - y))
        J = 1/(2*n) * np.matmul((np.matmul(X,self.theta) - y).T, (np.matmul(X,self.theta) - y))
        return J

    def gradient_descent(self, n, X, y):
        grad = 1/n * np.matmul(X.T, np.matmul(X, self.theta) - y)
        return grad
    
    def predict(self, testing_set):
        n = testing_set.shape[0]
        m = testing_set.shape[1]-1
        X = np.c_[(np.ones(n), testing_set[0:n, 0:m])]
        return 0
    
if __name__ == "__main__":
    # Load dataset
    training_data = pd.read_csv("dataset/l1_train.csv").to_numpy()
    testing_data = pd.read_csv("dataset/l1_test.csv").to_numpy()
    
    model = LinReg(epoch=10)
    model.fit(training_data)
    #print(model.predict(testing_data))
    
    
