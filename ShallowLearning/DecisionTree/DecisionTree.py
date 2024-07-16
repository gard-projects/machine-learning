import numpy as np

class Node:
    '''
    A class to represent a node in a decision tree.
    
    Attributes
    ----------
    feature_index : int
        The index of the feature that this node splits on.
    threshold : float
        The threshold value that this node splits on.
    left : Node
        The left child node.
    right : Node
        The right child node.
    value : int
        The predicted class label if this node is a leaf node.
    class_probs : np.ndarray
        The class probabilities if this node is a leaf node.
    '''
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, class_probs=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.class_probs = class_probs

class DecisionTree:
    '''
    A class to represent a decision tree.
    
    Attributes
    ----------
    root : Node
        The root node of the decision tree.
    
    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    max_features : int
        The number of features to consider when looking for the best split.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
        
    Methods
    -------
    fit(X, y)
        Constructs the decision tree, uses the _build_tree method.
    _build_tree(X, y, depth)
        Recursively builds the decision tree. Implements feature bagging.
    _get_feature_indices(n_features)
        Returns a list of feature indices to consider for the best split.
    _best_split(X, y, feature_indices)
        Finds the best feature and threshold to split on.
    _gini_impurity(y)
        Calculates the Gini impurity of a set of labels.
    predict(X)
        Predicts the class labels for a set of instances.
    predict_proba(X)
        Predicts the class probabilities for a set of instances.
    _tree_traversal(x, node)
        Traverses the decision tree to predict the class label of an instance, by fetching the value of the leaf node.
    _tree_traversal_proba(x, node)
        Traverses the decision tree to predict the class probabilities of an instance, by fetching the class probabilities of the leaf node.
    '''
    
    def __init__(self, max_depth=None, max_features=None, min_samples_split=2):
        '''
        Parameters
        ----------
        max_depth : int
            The maximum depth of the tree.
        max_features : int
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        '''
        
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : np.ndarray
            The training instances.
        y : np.ndarray
            The training labels.
        '''
        
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        '''
        Parameters
        ----------
        X : np.ndarray
            The training instances.
        y : np.ndarray
            The training labels.
    
        Returns
        -------
        Node object
            The root node of the decision tree.
        '''
        
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        class_counts = np.bincount(y, minlength=self.n_classes)
        class_probs = class_counts / len(y)

        # Stopping criteria (base case)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            return Node(value=np.argmax(class_counts), class_probs=class_probs)

        # Feature bagging
        feature_indices = self._get_feature_indices(n_features)

        best_feature_index, best_threshold = self._best_split(X, y, feature_indices)

        # If no split improves the criterion, create a leaf node
        if best_feature_index is None:
            return Node(value=np.argmax(class_counts), class_probs=class_probs)

        left_indices = X[:, best_feature_index] < best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold,
                    left=left_subtree, right=right_subtree, class_probs=class_probs)

    def _get_feature_indices(self, n_features):
        '''
        Parameters
        ----------
        n_features : int
            The number of features in the dataset.
        
        Returns
        -------
        list
            A list of feature indices to consider for the best split.
        '''
        
        if self.max_features is None:
            return range(n_features)
        else:
            return np.random.choice(n_features, size=self.max_features, replace=False)

    def _best_split(self, X, y, feature_indices):
        '''
        Parameters
        ----------
        X : np.ndarray
            The training instances.
        y : np.ndarray
            The training labels.
        feature_indices : list
            A list of feature indices to consider for the best split.
            
        Returns
        -------
        best_feature_index : int
            The index of the feature that gives the best split.
        best_threshold : float
            The corresponding threshold value.
        '''
        
        best_feature_index = None
        best_threshold = None
        best_gini_score = float('inf')

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = ~left_indices

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                gini_left = self._gini_impurity(y[left_indices])
                gini_right = self._gini_impurity(y[right_indices])
                gini_overall = (np.sum(left_indices) / len(y)) * gini_left + \
                               (np.sum(right_indices) / len(y)) * gini_right

                if gini_overall < best_gini_score:
                    best_gini_score = gini_overall
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _gini_impurity(self, y):
        '''
        Parameters
        ----------
        y : np.ndarray
            The labels of the instances.
        
        Returns
        -------
        gini_score : float
            The Gini impurity of the labels.
        '''
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : np.ndarray
            The instances to predict.
        
        Returns
        -------
        np.ndarray
            The predicted class labels.
        '''
        
        return np.array([self._tree_traversal(x, self.root) for x in X])

    def predict_proba(self, X):
        '''
        Parameters
        ----------
        X : np.ndarray
            The instances to predict.
            
        Returns
        -------
        np.ndarray
            The predicted class probabilities.
        '''
        
        return np.array([self._tree_traversal_proba(x, self.root) for x in X])

    def _tree_traversal(self, x, node):
        '''
        Parameters
        ----------
        x : np.ndarray
            The instance to predict.
        node : Node
            The current node in the traversal.
            
        Returns
        -------
        int
            The predicted class label, i.e. the value of the leaf node.
        '''
        
        if node.value is not None:
            return node.value

        if x[node.feature_index] < node.threshold:
            return self._tree_traversal(x, node.left)
        else:
            return self._tree_traversal(x, node.right)

    def _tree_traversal_proba(self, x, node):
        '''
        Parameters
        ----------
        x : np.ndarray
            The instance to predict.
        node : Node
            The current node in the traversal.
            
        Returns
        -------
        np.ndarray
            The predicted class probabilities, i.e. the class_probs of the leaf node.
        '''
        
        if node.value is not None:
            return node.class_probs

        if x[node.feature_index] < node.threshold:
            return self._tree_traversal_proba(x, node.left)
        else:
            return self._tree_traversal_proba(x, node.right)