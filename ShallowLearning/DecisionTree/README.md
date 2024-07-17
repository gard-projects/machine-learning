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

# Understanding the project
The project is about implementing RandomForest from scratch, and to do so we must also implement DecisionTree. The RandomForest classifier is essentially a collection of decision trees, but utilizies techniques like **bagging** and **random subspaces** to reduce the risk of overfitting. Why? Decision trees are prone to overfitting, especially when there are loose restrictions on the depth of the tree. However random forest combats this weakness, by providing each decision tree with a random subset of samples and features. Additionally, a random forest makes a prediction by providing the sample to be classified to all the decision trees within its scope, and makes a final decision based on the results of these weaker models.

A decision tree is a binary tree, but with its own set of rules. Each internal node has to be carefully selected, since each internal node represents a given feature (from the provided training set), and the best threshold for this feature.
My implementation makes use of a `Node` object, as it is easier to associate these attributes with a Python object rather than some n-dimensional array. 

I have decided to break down this guide into smaller parts, one part covering the DecisionTree &mdash; how it functions, and they way it is implemented &mdash; and how the RandomForest algorithm utilizes and interacts with the decision trees.



## Overview
This project implements RandomForest from scratch, which necessitates implementing DecisionTree as well. 

## Random Forest
RandomForest is an ensemble learning method that creates a collection of decision trees. It utilizes techniques like **bagging** (Bootstrap Aggregating) and **random subspaces** to reduce the risk of overfitting.

### Why use Random Forest?
Decision trees are prone to overfitting, especially with unrestricted depth. Random Forest combats this by:
1. Providing each decision tree with a random subset of samples (bagging)
2. Using a random subset of features for each tree (random subspaces)

### How Random Forest makes predictions
1. Presents the sample to all decision trees in the forest
2. Aggregates predictions from all trees (e.g., majority voting for classification)

## Decision Tree
A decision tree is a binary tree with specific rules:
- Each *internal node* represents a feature from the training set
- Each internal node has an optimal threshold for its feature
- *Leaf nodes* represent final predictions

### Implementation
We use a `Node` object to represent tree nodes:
