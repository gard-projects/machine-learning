import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from scipy.optimize import minimize_scalar
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

class GradientBoost:
    '''
    A class representing a Gradient Boosting Classifier.
    This is a type of ensemble algorithm that trains a series of weak learners sequentially, each learner correcting the errors of its predecessor.
    
    Parameters
    ----------
    n_estimators : int, default=50
        The number of weak learners to train.
    
    Attributes
    ----------
    n_estimators : int
        The number of weak learners to train.
    estimators_ : list
        A list of trained weak learners.
    learning_rate : list
        A list of learning rates for each weak learner.
        
    Methods
    -------
    find_gamma(X, y, F, h_m)
        Find the optimal learning rate for the weak learner, using SciPy's minimize_scalar.
    fit(X, y)
        Fit the model to the training data.
    predict_proba(X)
        Predict the probabilities of the classes.
    predict(X)
        Predict the classes.
    get_params(deep=True)
        Get the parameters of the model, needed for CV and GridSearch.
    set_params(**params)
        Set the parameters of the model, needed for CV and GridSearch.
    '''
    
    def __init__(self, n_estimators=50):
        '''
        Parameters
        ----------
        n_estimators : int, default=50
            The number of weak learners to train.
        
        Attributes
        ----------
        n_estimators : int  
            The number of weak learners to train.
        estimators_ : list
            A list of trained weak learners.
        '''
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.learning_rate = []

    def find_gamma(self, y, F, h_m):
        '''
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values.
        F : array-like of shape (n_samples,)
            The current predictions.
        h_m : array-like of shape (n_samples,)
            The predictions of the weak learner.
    
        Returns
        -------
        float
            The optimal learning rate for the weak learner.
        '''
        
        def objective_func(gamma):
            predictions = 1 / (1 + np.exp(-(F + gamma * h_m)))
            epsilon = 1e-10
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.sum(y * np.log(predictions) + (1-y) * np.log(1 - predictions))

        result = minimize_scalar(objective_func)
        return result.x

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) 
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self
            The GradientBoost object, fitted to the training data. Can be used for method chaining.
        '''
        
        p = np.sum(y == 1) / len(y)
        F = np.full((len(y),), np.log(p/ (1-p))) # F_0(x)
        
        for _ in range(self.n_estimators):
            y_pred = 1 / (1+np.exp(-F))
            grad = -(y - y_pred) # Pseudo-residual
            
            h_m = DecisionTreeRegressor().fit(X, grad)
            self.estimators_.append(h_m)
            h_m = h_m.predict(X)
            
            # Learning rate
            gamma = self.find_gamma(y, F, h_m)
            self.learning_rate.append(gamma)
            F += gamma * h_m      
        return self
           
    def predict_proba(self, X):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        array-like of shape (n_samples,)
            The predicted probabilities of the positive class.
        '''
        
        F = np.zeros(X.shape[0])
        for gamma, estimator in zip(self.learning_rate, self.estimators_):
            F += gamma * estimator.predict(X)
        return 1 / (1 + np.exp(-F))
    
    def predict(self, X):
        '''
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        array-like of shape (n_samples,)
            The predicted classes.
        '''
        
        probabilities = self.predict_proba(X)
        return np.where(probabilities > 0.5, 1, 0)
    
    def get_params(self, deep=True):
        '''
        Parameters
        ----------
        deep : bool, default=True
            Whether to return the parameters of the model.  
        
        Returns
        -------
        dict    
            The parameters of the model.
        '''
        
        return {'n_estimators': self.n_estimators}
    
    def set_params(self, **params):
        '''
        Parameters
        ----------
        **params
            The parameters to set in the model.
        
        Returns
        -------
        self
            The GradientBoost object, with the parameters set.
        '''
        self.n_estimators = params['n_estimators']
        return self


if __name__ == "__main__":
    data = pd.read_csv("dataset/stroke.csv")
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    
    cat_features = [1, 5, 6, 7, 10]
    num_features = [i for i in range(X.shape[1]) if i not in cat_features]
    
    # Data preprocessing
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    X = preprocessor.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Apply SMOTE to balance the classes (as there were way more 0s than 1s)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Pipelines
    c_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoost())
    ])

    sk_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier())
    ])
    
    c_result = c_model.fit(X_train, y_train).predict(X_test)
    sk_result = sk_model.fit(X_train, y_train).predict(X_test)
    print("\nCustom model: ", classification_report(y_test, c_result))
    print("\nSklearn model: ", classification_report(y_test, sk_result))
    
    c_conf = confusion_matrix(y_test, c_result)
    sk_conf = confusion_matrix(y_test, sk_result)
    ConfusionMatrixDisplay(c_conf).plot()
    ConfusionMatrixDisplay(sk_conf).plot()
    
    c_fpr, c_tpr, c_thresholds = roc_curve(y_test, c_result)
    c_auc = auc(c_fpr, c_tpr)
    sk_fpr, sk_tpr, sk_thresholds = roc_curve(y_test, sk_result)
    sk_auc = auc(sk_fpr, sk_tpr)
    RocCurveDisplay(fpr=c_fpr, tpr=c_tpr, roc_auc=c_auc, estimator_name="Custom model").plot()
    RocCurveDisplay(fpr=sk_fpr, tpr=sk_tpr, roc_auc=sk_auc, estimator_name="Sklearn model").plot()
    plt.show()
    