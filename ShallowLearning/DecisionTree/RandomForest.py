import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from DecisionTree import DecisionTree
from joblib import Parallel, delayed
from typing import List, Dict, Any
import matplotlib.pyplot as plt

class RandomForest:
    '''
    A class representing a Random Forest classifier.
    
    Attributes
    ----------
    trees : List[DecisionTree]
        A list of DecisionTree objects.
    
    Parameters
    ----------
    n_estimators : int
        The number of trees in the forest.
    max_depth : int
        The maximum depth of the trees.
    max_features : str
        The number of features to consider when looking for the best split.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    n_jobs : int
        The number of jobs to run in parallel.
    random_state : int
        The seed used by the random number generator.
        
    Methods
    -------
    fit(X, y)
        Fits the model to the training data, creating weak learners in the form of DecisionTree objects.
    _build_tree(X, y)
        Builds a single decision tree using bootstrap aggregation.
    predict_proba(X)
        Predicts the class probabilities for the input data.
    predict(X)
        Predicts the class labels for the input data.
    score(X, y)
        Returns the mean accuracy on the given test data and labels.
    get_params(deep)
        Returns the parameters of the estimator, needed for GridSearchCV and cross_val_score.
    set_params(**parameters)
        Sets the parameters of the estimator, needed for GridSearchCV and cross_val_score.
    '''
    
    def __init__(self, n_estimators: int, max_depth: int, max_features: str = 'sqrt',
                 min_samples_split: int = 2, n_jobs: int = -1, random_state: int = None):
        '''
        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the trees.
        max_features : str
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        n_jobs : int
            The number of jobs to run in parallel.
        random_state : int
            The seed used by the random number generator.
        '''
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trees: List[DecisionTree] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        '''
        Parameters
        ----------
        X : np.ndarray
            The training instances.
        y : np.ndarray
            The training labels.
            
        Returns
        -------
        self : object
            Estimator instance, allowing for method chaining.
        '''
        
        n_samples, n_features = X.shape
        
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(n_features))
        elif self.max_features is None:
            self.max_features = n_features
        
        np.random.seed(self.random_state)
        
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y) for _ in range(self.n_estimators)
        )
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray) -> DecisionTree:
        '''
        Parameters
        ----------
        X : np.ndarray
            The training instances.
        y : np.ndarray
            The training labels.
            
        Returns
        -------
        DecisionTree
            A DecisionTree object.
        '''
        
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample, y_sample = X[indices], y[indices]
        
        tree = DecisionTree(max_depth=self.max_depth, 
                            max_features=self.max_features, 
                            min_samples_split=self.min_samples_split)
        tree.fit(X_sample, y_sample)
        return tree

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        Parameters
        ----------
        X : np.ndarray
            The input data.
        
        Returns
        -------
        np.ndarray
            The class probabilities for the input data.
        '''
        
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Parameters
        ----------
        X : np.ndarray
            The input data.
        
        Returns
        -------
        np.ndarray
            The predicted class labels for the input data.
        '''
        
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target labels.
            
        Returns
        -------
        float
            The mean accuracy on the given test data and labels.
        '''
        return np.mean(self.predict(X) == y)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        '''
        Parameters
        ----------
        deep : bool
            Whether to return the parameters for this estimator and its subobjects.
        
        Returns
        -------
        dict: str
            Parameter names mapped to their values.
        '''
        
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'min_samples_split': self.min_samples_split,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state
        }

    def set_params(self, **parameters: Any) -> 'RandomForest':
        '''
        Parameters
        ----------
        **parameters : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        '''
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    
if __name__ == "__main__":
    data = pd.read_csv("dataset/bankloan.csv").to_numpy()
    # Standardize the features
    X = StandardScaler().fit_transform(data[:, :-1])
    y = data[:, -1].astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForest(n_estimators=100, max_depth=10, max_features='sqrt', min_samples_split=2, n_jobs=-1, random_state=42)
    rf_results = rf_model.fit(X_train, y_train.astype(int)).predict(X_test)
    sk_model = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', min_samples_split=2, n_jobs=-1, random_state=42)
    sk_results = sk_model.fit(X_train, y_train.astype(int)).predict(X_test)
    
    # Classification report, providing precision, recall, f1-score and support
    print("Custom Random Forest: \n", classification_report(y_test, rf_results))
    print("\n\n")
    print("Sklearn Random Forest: \n", classification_report(y_test, sk_results))
    
    # Display of confusion matrix
    rf_confusion_matrix = confusion_matrix(y_test, rf_results)
    sk_confusion_matrix = confusion_matrix(y_test, sk_results)
    rf_display = ConfusionMatrixDisplay(confusion_matrix=rf_confusion_matrix).plot()
    sk_display = ConfusionMatrixDisplay(confusion_matrix=sk_confusion_matrix).plot()

    # ROC curve and AUC score
    rf_fpr, rf_tpr, rf_thresh = roc_curve(y_test, rf_results)
    rf_auc = auc(rf_fpr, rf_tpr)
    rf_curve_display = RocCurveDisplay(fpr=rf_fpr, tpr=rf_tpr, roc_auc=rf_auc).plot()
    
    sk_fpr, sk_tpr, sk_thresh = roc_curve(y_test, sk_results)
    sk_auc = auc(sk_fpr, sk_tpr)
    sk_curve_display = RocCurveDisplay(fpr=sk_fpr, tpr=sk_tpr, roc_auc=sk_auc).plot()
    plt.show()
    