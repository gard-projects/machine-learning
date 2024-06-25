import pandas as pd
import numpy as np
import SMO as smo
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class SVM:
    '''
    A class that represents a custom estimator for a Support Vector Machine (SVM) model
    
    
    Attributes
    ----------
    max_iters : int
        Maximum number of iterations for the SMO algorithm
    tol : float
        Tolerance for the stopping criterion for the SMO algorithm
    C : float
        Regularization parameter
    support_ : np.ndarray
        Indices of the support vectors
    labels : np.ndarray
        Labels of the support vectors
    support_vectors : np.ndarray
        Support vectors containing the features from the training data
    n_support_ : int
        Number of support vectors
    alphas : np.ndarray
        Lagrange multipliers for the support vectors
        
        
    Methods
    -------
    get_params(deep=True)
        Returns the parameters of the estimator, needed for the GridSearchCV
    set_params(**parameters)
        Sets the parameters of the estimator, needed for the GridSearchCV
    fit(X, y)
        Fits the model to the training data by using the SMO algorithm
    decision_function(X)
        Calculates the decision function for the given data, used for prediction
    predict(test_data)
        Predicts the labels for the given data
    accuracy(test_data, labels)
        A method that calculates the accuracy of the model, in terms of correct predictions
    cross_validation(k, data)
        A method that performs k-fold cross validation on the given data, returns the mean accuracy
    '''
    def __init__(self, max_iters=1000, tol=0.01, C=1):
        '''
        Parameters
        ----------
        max_iters : int
            Maximum number of iterations for the SMO algorithm
        tol : float
            Tolerance for the stopping criterion for the SMO algorithm
        C : float
            Regularization parameter
        '''
        
        self.max_iters = 0
        self.tol = 0
        self.C = 0
        
    def get_params(self, deep=True):
        '''
        Parameters
        ----------
        deep : bool
            If True, will return the parameters of nested estimators
            
        Returns
        -------
        A dictionary containing the parameters of the estimator
        '''
        
        return {
            'max_iters': self.max_iters,
            'tol': self.tol,
            'C': self.C
        }
        
    def set_params(self, **parameters):
        '''
        Parameters
        ----------
        **parameters : dict
            A dictionary containing the parameters of the estimator
        
        Returns
        -------
        self : object
            Returns the estimator with the updated parameters, allows for method chaining
        '''
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : np.ndarray
            A matrix with n samples and m features, shape (n, m)
        y : np.ndarray
            A vector with n labels, shape (n, 1)
            
        Returns
        ------- 
        self : object
            Returns the estimator with the fitted model, allows for method chaining
        '''
        # Check X and y for complex data
        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError("Complex data not supported")
        
        alphas, self.b = smo.sequential_minimal_optimization(X, y, self.max_iters, self.tol, self.C)
        
        self.support_ = np.where(alphas > 0)[0]
        self.labels = y[self.support_]
        self.support_vectors = X[self.support_]
        self.n_support_ = len(self.support_)
        self.alphas = alphas[self.support_]
    
        return self
    
    def decision_function(self, X):
        '''
        Parameters
        ----------
        X : np.ndarray
            A matrix with n samples and m features, shape (n, m)
            
        Returns
        -------
        The decision function for the given data, shape (n, 1)
        '''
        
        kernel_func = smo.rbf_kernel
        K = kernel_func(X, self.support_vectors)

        return (np.dot(self.alphas*self.labels, K.T) + self.b)
    
    def predict(self, test_data):
        '''
        Parameters
        ----------
        test_data : np.ndarray
            A matrix with n samples and m features, shape (n, m)
        
        Returns
        -------
        The predicted labels for the given data, shape (n, 1)
        
        '''
        return np.sign(self.decision_function(test_data))
    
    def accuracy(self, test_data, labels):
        '''
        Parameters
        ----------
        test_data : np.ndarray
            A matrix with n samples and m features, shape (n, m)
        labels : np.ndarray
            A vector with n labels, shape (n, 1)
            
        Returns
        -------
        The accuracy of the model, in terms of correct predictions
        '''
        return np.mean(self.predict(test_data) == labels)
    
    def cross_validation(self, k, data: np.ndarray):
        '''
        Parameters
        ----------
        k : int
            Number of folds for the cross validation
        data : np.ndarray
            A matrix with n samples and m features, shape (n, m)
            
        Returns
        -------
        The mean accuracy of the model, in terms of correct predictions
        '''
        # Ten fold cross validation
        scores = []
        size = int(len(data)/k)
        target = data[:,-1]
    
        
        for i in range(k):
           test_data = data[i*size:(i+1)*size, :].astype(float)
           test_labels = target[i*size:(i+1)*size]
           train_data = np.concatenate([data[:i*size, :], data[(i+1)*size:, :]]).astype(float)
           train_labels = np.concatenate([target[:i*size], target[(i+1)*size:]])
                     
           mean = np.mean(train_data, axis=0)
           std = np.std(train_data, axis=0)
           train_data_std = (train_data - mean) / std
           test_data_std = (test_data - mean) / std          
                     
           self.fit(train_data_std, train_labels)
           score = self.accuracy(test_data_std, test_labels)
           scores.append(score)
        
        return np.mean(scores)

    
if __name__ == "__main__":
    data_excel = pd.read_excel("dataset/raisin.xlsx")
    data = data_excel.to_numpy()
    data[data == 'Kecimen'] = 1
    data[data == 'Besni'] = -1
    

    c_svm = SVM()
    param_grid1 = {
        'svm__max_iters': [1000],
        'svm__tol': [0.001],
        'svm__C': [2],
        'svm__gamma': [0.001]
    }
    
    grid1 = GridSearchCV(Pipeline([
        ('scaling', StandardScaler()),
        ('svm', SVM())
    ]), param_grid1, cv=10, n_jobs=-1, scoring='accuracy')
   
    
    grid1.fit(data[:,:-1].astype(float), data[:,-1].astype(int))
    print("Our implementation: ", grid1.best_score_)
    print("Number of support vectors:", grid1.best_estimator_.named_steps['svm'].n_support_)
    print(grid1.best_params_)
     
    # For linear kernel:    {'svm__C': 2, 'svm__max_iters': 200, 'svm__tol': 0.001}
    # For polynomial kernel:
    # For rbf kernel: {'svm__C': 2, 'svm__gamma': 0.001, 'svm__max_iters': 1000, 'svm__tol': 0.001}
        
    # Sklearn implementation of SVM (using SVC class)
    param_grid2 = {
        'svm__C':[1, 2, 4, 6, 8, 10], 
        'svm__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'svm__gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        }
    
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('svm', SVC(kernel='rbf'))
    ])

    grid2 = GridSearchCV(pipeline, param_grid2, cv=10, n_jobs=-1, scoring='accuracy')
    grid2.fit(data[:,:-1].astype(float), data[:,-1].astype(int))
    print()
    print("Sklearn implementation: ", grid2.best_score_)
    print("Number of support vectors:", grid2.best_estimator_.named_steps['svm'].n_support_)
    print("Best parameters: ", grid2.best_params_)