import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB as skMultinomialNB
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse


# Multinomial Naive Bayes

class MultinomialNB:
    '''
    Multinomial Naive Bayes classifier
    
    Attributes
    ----------
    alpha : float, default=1
        Laplace smoothing parameter
    classes_ : array-like of shape (n_classes,)
        The unique class labels
    class_count_ : array-like of shape (n_classes,)
        The number of samples for each class
    class_log_prior_ : array-like of shape (n_classes,)
        The log prior probability of each class
    word_count_ : array-like of shape (n_classes, n_features)
        The number of words for each class
    
    Methods
    -------
    fit(X, y)
        Fit the model according to the given training data
    predict(X, batch_size=1000)
        Perform classification on an array of test vectors X (using batch processing)
    log_likelihood(X)
        Compute the log likelihood of the samples X
    log_proba(X)
        Compute the probability estimates for the samples X
    score(X, y)
        Return the mean accuracy on the given test data and labels
    recall_score(X, y)
        Return the recall score on the given test data and labels
    precision_score(X, y)
        Return the precision score on the given test data and labels
    f1_score(X, y)
        Return the F1 score on the given test data and labels
    get_params(deep=True)
        Get parameters for this estimator (needed for Sklearn's cross-validation)
    set_params(**parameters)
        Set the parameters of this estimator (needed for Sklearn's cross-validation)
    '''
    
    def __init__(self, alpha=1):
        '''
        Parameters
        ----------
        alpha : float, default=1
            Laplace smoothing parameter
        '''
        
        self.alpha = alpha

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,)
            The target values (class labels)
        '''

        self.classes_ = np.unique(y)
        self.class_count_ = np.zeros(self.classes_.shape[0])
        self.class_log_prior_ = np.zeros(self.classes_.shape[0])
        self.word_count_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        
        for i, c in enumerate(self.classes_):
            c_k = (y == c)
            self.class_count_[i] = np.sum(c_k)
            self.class_log_prior_ = np.log((self.class_count_ + self.alpha) / (y.shape[0] + self.alpha * len(self.classes_))) # Laplace smoothing
            self.word_count_[i] = np.sum(X[c_k], axis=0)
    
    def predict(self, X, batch_size=1000):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        batch_size : int, default=1000
            The batch size for processing the input samples
            
        Returns
        -------
        array-like of shape (n_samples,)
            The predicted class labels (binary 0 or 1)
        '''
        predictions = np.zeros((X.shape[0]))
        
        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i+batch_size]
            batch_likelihood = self.log_likelihood(batch)
            predictions[i:i+batch_size] = self.classes_[np.argmax(batch_likelihood, axis=1)]
        
        return predictions
    
    def log_likelihood(self, X):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        
        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The log likelihood of the samples X (using matrix multiplication and sparse matrix handling)
        '''
        
        # Multinomial distribution
        b = self.class_log_prior_
        V = X.shape[1] # Vocabulary size
        w = np.zeros((self.classes_.shape[0], V))

        # Compute p_ki = log(P(x_i|y_k))
        for i in range(self.classes_.shape[0]):
            w[i] = np.log((self.word_count_[i] + self.alpha) / (np.sum(self.word_count_[i]) + self.alpha * V))
        
        # Check if sparse matrix, typically for text data
        if sparse.issparse(X):
            return b + X.dot(w.T)
        else:
            return b + np.dot(X, w.T)
        
    def log_proba(self, X):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The probability estimates for the samples X
        '''
        log_prob = self.log_likelihood(X)
        prob = np.exp(log_prob - np.max(log_prob, axis=1)[:, np.newaxis])
        return prob / np.sum(prob, axis=1)[:, np.newaxis]
         
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
    
    def recall_score(self, X, y):
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
            The specificity/recall of the model
        '''

        predictions = self.predict(X)
        tp = np.sum((predictions == 1) & (y == 1))
        fn = np.sum((predictions == 0) & (y == 1))
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def precision_score(self, X, y):
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
        
        predictions = self.predict(X)
        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        
        return tp / (tp + fp) if (tp + fp) > 0 else 0

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
        precision = self.precision_score(X, y)
        recall = self.recall_score(X, y)
        
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
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

        
if __name__ == "__main__":
    data_csv = pd.read_csv("dataset/emails.csv")
    data = CountVectorizer().fit_transform(data_csv["text"])
    #data = TfidfVectorizer().fit_transform(data_csv["text"])
    
    X_train, X_test, y_train, y_test = train_test_split(data, data_csv["spam"], test_size=0.2, random_state=42)
    c_model = MultinomialNB()
    sk_model = skMultinomialNB()
    
    c_scores = cross_val_score(c_model, X_train, y_train, cv=10, scoring="accuracy")
    sk_scores = cross_val_score(sk_model, X_train, y_train, cv=10, scoring="accuracy")
    print("---------- Cross validation ----------")
    print("Custom MultinomialNB: ", c_scores.mean())
    print("Scikit MultinomialNB: ", sk_scores.mean())
    
    
    print("\n\n---------- Evaluation (custom model) ----------")
    c_model.fit(X_train, y_train)
    print("Score of custom model: ", c_model.score(X_test, y_test))
    print("Recall: ", c_model.recall_score(X_test, y_test))
    print("Precision: ", c_model.precision_score(X_test, y_test))
    print("F1 Score: ", c_model.f1_score(X_test, y_test))
    
    print("\n\n---------- Evaluation (sklearn) ----------")
    sk_model.fit(X_train, y_train)
    print("Score of scikit model: ", sk_model.score(X_test, y_test))
    print("Recall: ", recall_score(y_test, sk_model.predict(X_test)))
    print("Precision: ", precision_score(y_test, sk_model.predict(X_test)))
    print("F1 Score: ", f1_score(y_test, sk_model.predict(X_test)))
    
    plot_confusion_matrix(y_test, c_model.predict(X_test))

    
    