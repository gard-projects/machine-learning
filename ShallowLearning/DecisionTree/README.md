**DATASET FROM** \
https://www.kaggle.com/datasets/vikramamin/bank-loan-approval-lr-dt-rf-and-auc


# Structure of the dataset
The dataset contains 14 columns, and 5000 samples. Thus the shape of the `data` variable is $\left(5000, 14\right)$. 
> **ID**, type=int64
>
> **Age**, type=int64, predictor variable
>
> **Experience**, type=int64, predictor variable
>
> **Income**, type=int64, predictor variable
>
> **ZIP.Code**, type=int64, predictor variable
>
> **Family**, type=int64, predictor variable
>
> **CCAvg**, type=float64, predictor variable
>
> **Education**, type=int64, predictor variable
>
> **Mortgage**, type=int64, predictor variable
>
> **Personal.Loan**, type=int64, response variable
>
> **Securities.Account**, type=int64, predictor variable
>
> **CD.Account**, type=int64, predictor variable
>
> **Online**, type=int64, predictor variable
>
> **CreditCard**, type=int64, predictor variable

# Overview
This project focuses on implementing RandomForest and DecisionTree from scratch. RandomForest is considered as one of the best machine learning models for shallow learning, it is also known as a **discriminative** machine learning model.
We make use of this model to predict whether a given person is eligible for a personal loan given the associated features. 

## Decision Tree
A decision tree is similar to a binary tree in computer sience, but has its own set of rules.
1. Each **internal node** represents a feature from the provided training set
2. Each internal node has an optimal threshould for its feature
3. **Leaf nodes** represent the final predictions

A decision tree has various parameters, such as:
> **max_depth**: the maximum depth of a decision tree
>
> **max_features**: the maximum number of features to be considered when selecting a feature for an internal node
>
> **min_samples_split**: the minimum number of samples required to perform a split

### How does it work? 



1. Intialize the root node
2. Recursively populate the tree with the `fit()`and `_build_tree()` functions
3. Check if the constraints are violated: \
:one: The current depth is larger or equal to the maximum depth \
:two: The maximum depth provided to the given node is **None** \
:three: The number of samples provided to the node (by checking the shape of `X`) is smaller than the minimum number of samples for a split to be applied \
:four: The number of labels (from the target array `y`) is equal to 1 
```
    if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            return Node(value=np.argmax(class_counts), class_probs=class_probs)
```
4. For each node, fetch the features provided in the training set `X` using the function `_get_feature_indices`
```
feature_indices = self._get_feature_indices(n_features)
```
6. Find the best feature and threshold that should be set on the given internal node, by using the `_best_split()` function
```
best_feature_index, best_threshold = self._best_split(X, y, feature_indices)

# If no split improves the criterion, create a leaf node
if best_feature_index is None:
    return Node(value=np.argmax(class_counts), class_probs=class_probs)
```
7. To fetch the best feature for the current depth, we use a metric called Gini impurity. Select the feature with the lowest Gini impurity score.

$$I_{G}(p) = \sum_{i=1}^{J} p_{i}(1-p_{i}) = \quad \sum_{i=1}^{J}p_{i} - \sum_{i=1}^{J} p_{i}^{2} = \quad 1 - \sum_{i=1}^{J} p_{i}^{2}$$

8. Once a leaf node is reached, set the `value` parameter of the Node object, and the `class_probs` parameter
