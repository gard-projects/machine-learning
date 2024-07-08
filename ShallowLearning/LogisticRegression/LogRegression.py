import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay

class LogRegression:
    '''
    A custom class used to represent a Logistic Regression model
    
    Attributes
    ----------
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations taken for the solver to converge
    learning_rate : float
        The learning rate for the gradient descent algorithm
    w : np.ndarray
        The weights of the model wtih a shape (number_features + 1,), includes the bias term (being the first element)
        
    Methods
    -------
    fit(X, y)
        Fits the model to the training data
    sigmoid(z)
        Calculates the sigmoid of the input (probability)
    predict(X)
        Predicts the class labels for the given data
    score(X, y)
        Returns the mean accuracy on the given test data and labels
    get_params(deep=True)
        Returns the parameters of the estimator, needed for Sklearn's Cross Validation and GridSearchCV
    set_params(**parameters)
        Sets the parameters of the estimator, needed for Sklearn's Cross Validation and GridSearchCV
    '''
    def __init__(self, tol=0.0001, max_iter=100, learning_rate=0.01):
        '''
        Parameters
        ----------
        tol : float, default=0.0001
            Tolerance for stopping criteria
        max_iter : int, default=100
            Maximum number of iterations taken for the solver to converge
        learning_rate : float, default=0.01
            The learning rate for the gradient descent algorithm
        '''
        
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + 1)
            Training samples
        y : array-like, shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns the instance itself, allows for method chaining
        '''
        
        self.w = np.random.rand(X.shape[1])
        
        for i in range(self.max_iter): 
            z = np.dot(X, self.w)
            y_pred = self.sigmoid(z)
            
            # Gradient Descent
            grad = np.dot(X.T, (y - y_pred))
            if np.all(np.abs(grad) < self.tol):
                break # Convergence
            self.w = self.w + self.learning_rate * grad
        return self
    
    def sigmoid(self, z):
        '''
        Parameters
        ----------
        z : array-like
            The linear combination of the weights and the input features (logits)
            
        Returns
        -------
        array-like
            The sigmoid of the input
        '''
        
        return 1 / (1+np.exp(-z))
    
    def predict(self, X):
        '''
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + 1)
            Test samples
        
        Returns
        -------
        array-like, shape (n_samples,)
            Predicted class label per sample
        '''
        
        treshold = 0.5
        return np.where(self.sigmoid(np.dot(X, self.w)) > treshold, 1, 0)

    def score(self, X, y):
        '''
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + 1)
            Test samples
        y : array-like, shape (n_samples,)
            True labels for X
            
        Returns
        -------
        float
            Returns the mean accuracy on the given test data and labels
        '''
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

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
            'tol': self.tol,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate
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
    

if __name__ == "__main__":
    data = pd.read_csv("dataset/framingham.csv").to_numpy()
    # TenYearCHD (response variable)
    
    # x = (4238, 15), X = (4238, 16), and y = (4238,)
    x = data[:, :-1]
    y = data[:, -1]
    
    # IMPORTANT: Impute missing values (NaN)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = imp.fit_transform(x)
    # Normalize the features
    x = StandardScaler().fit_transform(x)
        
    # Design matrix
    X = np.column_stack((np.ones((x.shape[0], 1)), x))
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    c_model = LogRegression(tol=0.0001, max_iter=100, learning_rate=0.001)
    sk_model = LogisticRegression(max_iter=100, tol=0.0001)

    # params = {
    #     'tol': [0.0001, 0.001, 0.01],
    #     'max_iter': [100, 200, 300, 1000],
    #     'learning_rate': [0.1, 0.01, 0.001, 0.0001]
    # }
    
    # clf = GridSearchCV(c_model, params, cv=10, n_jobs=-1).fit(x_train, y_train)
    # print(clf.best_params_) 
    # Gives: learning_rate=0.001, max_iter=100, tol=0.0001


    #  Evaluation (Cross Validation)
    print("------------ Cross Validation ------------")
    c_scores = cross_val_score(c_model, x_train, y_train, cv=10)
    print("Custom model: ", np.mean(c_scores))
    sk_scores = cross_val_score(sk_model, x_train, y_train, cv=10)
    print("Sklearn model: ", np.mean(sk_scores))
    
    # Fit model, predict and evaluate (test set)
    c_results = c_model.fit(x_train, y_train).predict(x_test)
    sk_results = sk_model.fit(x_train, y_train).predict(x_test)
    # Classification report
    print(classification_report(y_test, c_results, labels=[0, 1]))
    print("\n")
    print(classification_report(y_test, sk_results, labels=[0, 1]))
    
    # Confusion matrix
    c_confusion = confusion_matrix(y_test, c_results, labels=[0, 1])
    c_disp = ConfusionMatrixDisplay(confusion_matrix=c_confusion, display_labels=["Has TenYearCHD", "Not TenYearCHD"])
    sk_confusion = confusion_matrix(y_test, sk_results, labels=[0, 1])
    sk_disp = ConfusionMatrixDisplay(confusion_matrix=sk_confusion, display_labels=["Has TenYearCHD", "Not TenYearCHD"])
    c_disp.plot()
    sk_disp.plot()

    
    # ROC curve and AUC score
    c_fpr, c_tpr, c_tresholds = roc_curve(y_test, c_results)
    c_roc_auc = auc(c_fpr, c_tpr)
    c_roc_auc_disp = RocCurveDisplay(fpr=c_fpr, tpr=c_tpr, roc_auc=c_roc_auc, estimator_name="Custom Model")
    
    sk_fpr, sk_tpr, sk_tresholds = roc_curve(y_test, sk_results)
    sk_roc_auc = auc(sk_fpr, sk_tpr)
    sk_roc_auc_disp = RocCurveDisplay(fpr=sk_fpr, tpr=sk_tpr, roc_auc=sk_roc_auc, estimator_name="Sklearn Model")
    
    c_roc_auc_disp.plot()
    sk_roc_auc_disp.plot()
    plt.show()