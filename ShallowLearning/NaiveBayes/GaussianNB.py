import pandas as pd
import numpy as np


class GaussianNB():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.theta_ = np.zeros((len(self.classes_), X.shape[1])) # Mean, kxN matrix
        self.var_ = np.zeros((len(self.classes_), X.shape[1])) # Variance, kxN matrix
        self.class_prior_ = np.zeros(len(self.classes_)) # Prior probability N-dimensional vector
        
        for k in range(X.shape[1]):
            class_mask = (y == k)
            class_features = X[class_mask]
            self.theta_[k] = np.mean(class_features, axis=0)
            self.var_[k] = np.var(class_features, axis=0)
            self.class_prior_[k] = self.get_prior_probability(y, k)
        
        return 0
    
    def gaussian_probability(self, v, mean, std):
        density = (1/np.sqrt(2*np.pi*std**2))*np.exp(-((v-mean)**2 / (2*std**2)))
        return density
    
    def get_prior_probability(self, y, k):
        return np.sum(y == k) / len(y)

if __name__ == "__main__":
    data = pd.read_csv("dataset/diabetes.csv")
    
    # 995 samples, 2 features (both of type int64), 1 target (type int64)
    # Two classes: "has diabetes" and "does not have diabetes" (indicated by 1 and 0)
    data = data.to_numpy()
    X = data[:,:-1]
    y = data[:, -1]
    
    c_model = GaussianNB()
    c_model.fit(X, y)
    
    #TODO:
    # Predict function
    # Log Likelihood function (prevent underflow probabilities)
    # Laplace smoothing