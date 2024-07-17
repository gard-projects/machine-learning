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
