import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB as skGaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class GaussianNB():
    '''
    Gaussian Naive Bayes classifier
    
    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        The classes labels (0 and 1)
    theta_ : 
        array of shape (n_classes, n_features), where each row is the mean of the features for the given class
    var_ :  
        array of shape (n_classes, n_features), where each row is the variance of the features for the given class
    class_prior_ :
        array of shape (n_classes,), where each element is the prior probability of the class
    
    Methods
    -------
    fit(X, y) : Fit the model according to the given training data
    predict(X) : Perform classification on an array of test vectors X
    log_proba(X) : Return the log-probability estimates for the test vector X
    predict_proba(X) : Return probability estimates for the test vector X
    '''
    
    def __init__(self, var_smoothing=1e-9, alpha=1.0):
        '''
        Parameters
        ----------
        var_smoothing : float, default=1e-9
            Used to reduce the variance of the features, prevents division by zero
        alpha : float, default=1.0
            Smoothing parameter, used to prevent division by zero when calculating class prior (Laplace smoothing)
        '''
        
        self.var_smoothing = var_smoothing
        self.alpha = alpha
    
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples
        y : array-like of shape (n_samples,)
            The target values (class labels)
        '''
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes) 
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            
            self.theta_[i] = X_c.mean(axis=0)
            self.var_[i] = X_c.var(axis=0) + self.var_smoothing # Variance smoothing
            self.class_prior_[i] = X_c.shape[0] / n_samples

            self.class_prior_[i] = (n_c + self.alpha) / (n_samples + self.alpha * n_classes) # Laplace smoothing
    
    
    def gaussian_log_probability(self, X, mean, var):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        mean : array-like of shape (n_features,)
            The mean of the features for the given class
        var : array-like of shape (n_features,)
            The variance of the features for the given class
            
        Returns
        ------- 
        log probability of the input samples
        '''
        
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2 / var)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        array of shape (n_samples,)
            The predicted class labels
        '''
        
        return np.argmax(self.log_proba(X), axis=1)
    
    def log_proba(self, X):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        array of shape (n_samples, n_classes)
            The log-probability estimates for the input samples
        '''
        
        n_samples, n_features = X.shape
        log_proba = np.zeros((n_samples, len(self.classes_)))
        
        for i, _ in enumerate(self.classes_):
            prior = np.log(self.class_prior_[i])
            likelihood = np.sum(self.gaussian_log_probability(X, self.theta_[i], self.var_[i]), axis=1)
            log_proba[:, i] = prior + likelihood
        
        return log_proba

    def predict_proba(self, X):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        array of shape (n_samples, n_classes)
            The probability estimates for the input samples
        '''
        
        log_prob = self.log_proba(X)
        prob = np.exp(log_prob - np.max(log_prob, axis=1, keepdims=True))
        return prob / np.sum(prob, axis=1, keepdims=True)
    
    def score(self, X, y):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,)
            The target values (class labels)
            
        Returns
        -------
        float :
            The mean accuracy of the model
        '''
        return np.mean(self.predict(X) == y)
    
    def specificity(self, X, y):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,)
            The target values (class labels)
            
        Returns
        -------
        float :
            The specificity of the model
        '''
        # Let True Negative = 0 (no diabetes)
        predictions = self.predict(X)
        
        tn = np.sum((predictions == 0) & (y == 0))
        fp = np.sum((predictions == 1) & (y == 0))
        
        if(tn + fp == 0):
            return 0
        
        return tn / (tn + fp)
    
    def sensitivity(self, X, y):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,)
            The target values (class labels)
        
        Returns
        -------
        float :
            The sensitivity of the model
        '''
        # Let True Positive = 1 (has diabetes)
        predictions = self.predict(X)
        
        tp = np.sum((predictions == 1) & (y == 1))
        fn = np.sum((predictions == 0) & (y == 1))
        
        if tp + fn == 0:
            return 0
        
        return tp / (tp + fn)

    def precision(self, X, y):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,)
            The target values (class labels)
            
        Returns
        -------
        float :
            The precision of the model
        '''
        # Let True Positive = 1 (has diabetes)
        predictions = self.predict(X)
        
        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        
        if tp + fp == 0:
            return 0
        
        return tp / (tp + fp)

    def f1_score(self, X, y):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,)
            The target values (class labels)
        
        Returns
        -------
        float :
            The F1 score of the model  
        '''
        precision = self.precision(X, y)
        recall = self.sensitivity(X, y)
        
        if precision + recall == 0:
            return 0
        
        return 2 * (precision * recall) / (precision + recall)

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
            'var_smoothing': self.var_smoothing,
            'alpha': self.alpha
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

def plot_confusion_matrix(y_true, y_pred):
    '''
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values
    y_pred : array-like of shape (n_samples,)
        The predicted target values
    '''
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("dataset/diabetes.csv")
    
    # 995 samples, 2 features (both of type int64), 1 target (type int64)
    # Two classes: "has diabetes" and "does not have diabetes" (indicated by 1 and 0)
    data = data.to_numpy()
    X = data[:,:-1]
    y = data[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    c_model = GaussianNB()
    sk_model = skGaussianNB()
    
    # Cross validation
    print()
    print("---------- Cross validation ----------")
    c_scores = cross_val_score(c_model, X_train, y_train, cv=10, scoring='accuracy')
    print(f"Mean accuracy (custom model): {c_scores.mean().round(6)}")
    sk_scores = cross_val_score(sk_model, X_train, y_train, cv=10, scoring='accuracy')
    print(f"Mean accuracy (sklearn): {sk_scores.mean().round(6)}")
    
    # Final modeling, and evaluation (on test data)
    c_model.fit(X_train, y_train)
    sk_model.fit(X_train, y_train)
    
    print("\n\n---------- Evaluation (custom model) ----------")
    print(f"Accuracy (custom model): {c_model.score(X_test, y_test).round(6)}")
    print(f"Specificity (custom model): {c_model.specificity(X_test, y_test).round(6)}")
    print(f"Sensitivity (custom model): {c_model.sensitivity(X_test, y_test).round(6)}")
    print(f"Precision (custom model): {c_model.precision(X_test, y_test).round(6)}")
    print(f"F1 Score (custom model): {c_model.f1_score(X_test, y_test).round(6)}")
    
    print("\n\n---------- Evaluation (sklearn) ----------")
    print(f"Accuracy (sklearn): {round(sk_model.score(X_test, y_test), 6)}")
    confusion_matrix = metrics.confusion_matrix(y_test, sk_model.predict(X_test).round(6))
    print(f"Specificity (sklearn): {(confusion_matrix[0, 0] / np.sum(confusion_matrix[0])).round(6)}")
    print(f"Sensitivity (sklearn): {metrics.recall_score(y_test, sk_model.predict(X_test)).round(6)}")
    print(f"Precision (sklearn): {metrics.precision_score(y_test, sk_model.predict(X_test)).round(6)}")
    print(f"F1 Score (sklearn): {metrics.f1_score(y_test, sk_model.predict(X_test)).round(6)}")

    plot_confusion_matrix(y_test, c_model.predict(X_test))