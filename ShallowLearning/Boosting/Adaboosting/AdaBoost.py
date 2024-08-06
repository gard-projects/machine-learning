import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt



class DecisionStump:
    '''
    A class representing a decision stump, a decision tree with a single split. Also called a weak learner in the context of boosting.
   
    Attributes
    ----------
    n_classes : np.ndarray
        The unique classes in the target variable
    feature_idx : int
        The index of the feature that the stump splits on
    threshold : float
        The threshold value for the feature
    left_scores : np.ndarray
        The confidence scores for the left split
    right_scores : np.ndarray
        The confidence scores for the right split
        
    Methods
    -------
    fit(X, y, w)
        Fits the stump to the training data
    weighted_entropy(y, w)
        Calculates the weighted entropy of the target variable
    confidence_score(y, w)
        Calculates the confidence scores of the target variable
    predict(X)
        Predicts the target variable for the given data
    '''
    
    def __init__(self):
        '''
        Attributes
        ----------
        learning_rate : float
            The learning rate of the weak learner, representing the importance of the learner in the ensemble   
        '''
        self.learning_rate = 0

    
    def fit(self, X, y, w):
        '''
        Parameters
        ----------
        X : np.ndarray
            The feature matrix
        y : np.ndarray
            The target variable
        w : np.ndarray
            The weights of the samples

        Returns
        -------
        self
            A fitted instance of the DecisionStump class, allowing for method chaining
        '''
        
        n_samples, n_features = X.shape
        self.n_classes, _ = np.unique(y, return_counts=True)
        parent_entropy = self.weighted_entropy(y, w)
        
        best_ig_score = float('-inf')
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
               
                y_left, w_left = y[left_mask], w[left_mask]
                y_right, w_right = y[right_mask], w[right_mask]
                
                w_left_total, w_right_total = np.sum(w_left), np.sum(w_right)
                w_total = w_left_total + w_right_total
                
                left_entropy = self.weighted_entropy(y_left, w_left)
                right_entropy = self.weighted_entropy(y_right, w_right)
                
                weighted_avg_entropy = (w_left_total / w_total) * left_entropy + \
                                       (w_right_total / w_total) * right_entropy
                information_gain = parent_entropy - weighted_avg_entropy
                
                if information_gain > best_ig_score:
                    self.feature_idx = feature
                    self.threshold = threshold
                    best_ig_score = information_gain
                    self.left_scores = self.confidence_score(y_left, w_left)
                    self.right_scores = self.confidence_score(y_right, w_right)
        return self
     
    def weighted_entropy(self, y, w):
        '''
        Parameters
        ----------
        y : np.ndarray
            The target variable
        w : np.ndarray
            The weights of the samples
            
        Returns
        -------
        float
            The weighted entropy of the target variable
        '''
        
        scores = self.confidence_score(y, w)      
        return -np.sum(scores * np.log2(scores + 1e-10))
    
    def confidence_score(self, y, w):
        '''
        Parameters
        ----------
        y : np.ndarray
            The target variable
        w : np.ndarray
            The weights of the samples
            
        Returns
        -------
        np.ndarray
            The confidence scores of the target variable    
        '''
        
        w_total = np.sum(w)
        w_class = np.array([np.sum(w[y==c]) for c in self.n_classes])    
        conf_score = w_class / (w_total+1e-10)     
        return conf_score
    
    def predict(self, X):
        '''
        Parameters
        ----------
        X : np.ndarray
            The feature matrix
            
        Returns
        -------
        np.ndarray
            The predicted target variable
        '''
        
        mask = X[:, self.feature_idx] <= self.threshold
        predictions = np.where(mask[:, np.newaxis], self.left_scores, self.right_scores)
        return predictions


class AdaBoostModel:
    '''
    A class representing the Adaboost classifier, an ensemble method that fits a sequence of weak learners on the training data.
    It does so by assigning weights to the samples, and updating the weights based on classification errors.
    
    Attributes
    ----------
    n_estimators : int
        The number of weak learners in the ensemble
    estimators : list
        A list of fitted weak learners
        
    Methods
    -------
    fit(X_train, y_train)
        Fits the ensemble to the training data
    weighted_error(y_true, y_pred, w)
        Calculates the weighted error of the predictions
    predict(X_test)
        Predicts the target variable for the given data
    get_params(deep=True)
        Returns the parameters of the model, needed for CV and GridSearch
    set_params(**params)
        Sets the parameters of the model, needed for CV and GridSearch
    '''
    
    def __init__(self, n_estimators=50):
        '''
        Parameters
        ----------
        n_estimators : int, default=50
            The number of weak learners in the ensemble
            
        Attributes
        ----------
        estimators : list
            A list of fitted weak learners
        '''
        
        self.n_estimators = n_estimators
        self.estimators = []
        
    def fit(self, X_train, y_train): 
        '''
        Parameters
        ----------
        X_train : np.ndarray
            The feature matrix
        y_train : np.ndarray
            The target variable
            
        Returns
        -------
        self
            A fitted instance of the AdaBoostModel class, allowing for method chaining
        '''
                
        n_samples = X_train.shape[0]
        w = np.full((n_samples), 1/n_samples)

        for _ in range(self.n_estimators):
            estimator = DecisionStump()
            estimator.fit(X_train, y_train, w)
            self.estimators.append(estimator)
            y_pred = np.argmax(estimator.predict(X_train), axis=1) # Majority voting
            err = self.weighted_error(y_train, y_pred, w) 

            K = len(estimator.n_classes)
            alpha = np.log((1-err) / err) + np.log(K - 1)
            estimator.learning_rate = alpha
             
            if alpha > 0 and err < 1 - 1/K:
                w = w * np.exp(alpha * (y_train != y_pred))
                w = w / np.sum(w) # Re-normalize
        return self
        
    def weighted_error(self, y_true, y_pred, w):
        '''
        Parameters
        ----------
        y_true : np.ndarray
            The true target variable
        y_pred : np.ndarray
            The predicted target variable
        w : np.ndarray
            The weights of the samples
            
        Returns
        -------
        float
            The weighted error of the predictions
        '''
        
        misclassified = (y_true != y_pred)
        error = np.sum(w[misclassified]) / np.sum(w)    
        return error
    
    def predict(self, X_test):
        '''
        Parameters
        ----------
        X_test : np.ndarray
            The feature matrix
        
        Returns
        -------
        np.ndarray
            The predicted target variable
        '''
        
        predictions = np.sum([e.learning_rate * e.predict(X_test) for e in self.estimators], axis=0)
        return np.argmax(predictions, axis=1)
    
    def get_params(self, deep=True):
        '''
        Parameters
        ----------
        deep : bool, default=True
            Whether to return the parameters of the model
            
        Returns
        -------
        dict
            The parameters of the model
        '''
        
        return {'n_estimators': self.n_estimators}
    
    def set_params(self, **params):
        '''
        Parameters
        ----------
        **params
            The parameters of the model
        
        Returns
        -------
        self
            An instance of the AdaBoostModel class with the updated parameters
        '''
        
        self.n_estimators = params['n_estimators']
        return self
    
if __name__ == "__main__":
    data = pd.read_csv("dataset/studentPerformance.csv").to_numpy()
    X = StandardScaler().fit_transform(X=data[:, :-1])
    y = data[:,-1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    # Cross validation
    c_model = AdaBoostModel(n_estimators=50)
    sk_model = AdaBoostClassifier(n_estimators=50, algorithm='SAMME')
    
    # param_grid = {'n_estimators': [10, 50, 100, 200]}
    # c_grid = GridSearchCV(estimator=c_model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=5).fit(X_train, y_train)
    # print(c_grid.best_params_)
    # print(c_grid.best_score_)

    # c_cv = cross_val_score(estimator=c_model, X=X_train, y=y_train, scoring='accuracy')
    # sk_cv = cross_val_score(estimator=sk_model, X=X_train, y=y_train, scoring='accuracy')
    # print(f"Custom CV: {c_cv.mean()}")
    # print(f"Sklearn CV: {sk_cv.mean()}")

    c_model.fit(X_train, y_train)
    sk_model.fit(X_train, y_train)
    
    c_results = c_model.predict(X_test)
    sk_results = sk_model.predict(X_test)
    print("\n Custom model: \n", classification_report(y_test, c_results))
    print("\n Sklearn model: \n", classification_report(y_test, sk_results))
    
    ConfusionMatrixDisplay(confusion_matrix(y_test, c_results)).plot()
    ConfusionMatrixDisplay(confusion_matrix(y_test, sk_results)).plot()
    plt.show()
    

