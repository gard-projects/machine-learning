> [!NOTE]
> Dataset used in this project can be found at: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset?resource=download

# Project description
This project is about implementing KNN from scratch. K-Nearest Neighbors (KNN) is a type of algorithm in machine learning used for classification and regression problems.
In this project I implemented KNN for the purpose of classification. The dataset used can be found in the dataset folder under the name `adult.csv`. The dataset contains the following:

> 15 variables:  9 of which are categorical, 6 being numerical 
> 
> **Categorical** variables: work-class, education, martial-status, occupation, relationship, race, gender, native-country, income 
> 
> **Numerical** variables: age, fnlwgt, educational-num, capital-gain, capital-loss, hours-per-week

We want to classify whether the income of a given individual is above 50K or less than 50K. Thus, `income` is the **dependent variable** in our dataset. 

# Assumptions and choises made 
The first assumption made is that the numerical variables are normally distributed, therefore we apply standardization to improve model performance. See below.
```
 numerical_features = [0, 2, 4, 10, 11, 12]
self.mean = np.mean(self.X_features[:, numerical_features].astype(float), axis=0)
self.std = np.std(self.X_features[:, numerical_features].astype(float), axis=0)
        
self.X_features[:, numerical_features] = (self.X_features[:, numerical_features] - self.mean) / self.std
self.testing_set[:, numerical_features] = (self.testing_set[:, numerical_features] - self.mean) / self.std
```
I chose to apply target encoding to education, occupation and native-country as each of these variables have a good amount of categories.
> Education = 16 unique categories
> 
> Occupation = 15 unique categories
> 
> Native-country = 42 unique categories

We use **target encoding** for `education`, `occupation`, and `native-country` due to their large number of categories (16, 15, and 42 respectively). This reduces dimensionality compared to **one-hot encoding**, which is used for other categorical variables with fewer categories. **Label encoding** is applied to the dependent variable since it has only two categories.

Batch processing has been used in this project due to a relatively large dataset. The reason is simple, due to the increased dimensionality of each `dist-batch`, I ran out of memory when running the model. Thus, I made us of batch processing to be able to properly compute the distances between each data point in testing set to all data points in the training set.
```
dist_batch = np.sqrt(np.sum((x1_batch[np.newaxis, :, :].astype(float) - x2_batch[:, np.newaxis, :].astype(float))**2,axis=2))
distances[i:end_i_index, j:end_j_index] = dist_batch.T
```

> [!NOTE]
> The reason we need to increase the dimension is due to the fact that Numpy is not able to broadcast the first axis (of x1) to match the first axis of x2 array. By adding a new dimension, we can effectively account for broadcasting to allow the distances from each batch to be stored in the `distances` numpy array.


# Main body of the KNN algorithm
The algorithm is divided into two main parts: `fit()` and `predict()`. The `fit()` function encodes the training and testing sets, while `predict()` computes distances and finds the K nearest neighbors. The algorithm sorts these distances and uses majority voting to classify the test points:

The KNN algorithm sorts this returned numpy arrays, and chooses the k nearest data points, in our example k is set to 5. Since k = 5, we retrieve the 5 closest points and look at the labels of these points.
The category with the most data points out of the 5 is chosen as the estimated classification of a given point. This is explicitly highlighted in the code below:

```
neighbors = np.argsort(distances, axis=1)[:, :self.k]
neighbor_labels = self.y[neighbors]
for i in range(testing_set.shape[0]):
  unique_labels, counts = np.unique(neighbor_labels[i,:], return_counts=True)
  majority_vote = unique_labels[np.argmax(counts)]
  predicted_response.append(majority_vote)
```
Once all samples in testing set is classified, we compute the accuracy of the model given by the function `accuracy()`. Here are two examples, first example using Euclidean distance and the second using Manhattan distance formula.
