DATASET: https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset/data

# The structure of the dataset
This dataset has multiple classes, thus we are dealing with **multiclass classification**. 
> **StudentID**, type=int, predictor variable
>
> **Age**, type=int, predictor variable
>
> **Gender**, type=int, predictor variable
>
> **Ethnicity**, type=int, predictor variable
>
> **ParentalEducation**, type=int, predictor variable
>
> **StudyTimeWeekly**. type=float, predictor variable
>
> **Absences**, type=int, predictor variable
>
> **Tutoring**, type=int, predictor variable
>
> **ParentalSupport**, type=int, predictor variable
>
> **Extracurricular**, type=int, predictor variable
>
> **Sports**, type=int, predictor variable
>
> **Music**, type=int, predictor variable
>
> **Volunterring**, type=int, predictor variable
>
> **GPA**, type=float, predictor variable
>
> **GradeClass**, type=int, **response variable**

As you can see this dataset has already done encoding on the categorical features, thus we require few preprocessing steps before training the model.

# Decision Stump
A decision stump is in simple terms a decision tree which consists of a root node, and two leaf nodes. In other words, you only need to determine the best feature once, then split the data according to this feature and its corresponding threshold.
Adaboost typically uses decision stumps &emdash; unlike GradientBoosting which uses decision trees with greater depths &emdash; as weak learners. Here are some reasons as to why.

* Weak learners should perform slightly better than random guessing, decision stumps satisfy this criteria
* Decision trees with great depths are more prone to overfitting, which is not the case for decision stumps
* Decision stumps can be trained very quickly


# Information gain and entropy
To find the best feature to split on when training a decision tree, you can choose between various metrics such as **Gini impurity** or **Information gain**. Do note that there exists many other techniques, the ones mentioned only a few.
In this project we will be using information gain, which uses a concept called **entropy**. Information gain, like Gini impurity, tells us how important a given feature is. However unlike Gini impurity &emdash; in which lower scores is better &emdash; features with greater information gain are preferred. 

$$\text{Entropy, e} = \quad - \sum_{i=1}^{n} p_i \log_{2}(p_i)$$

Information gain is given by the following equation

$$\text{IG}(T, a) = \quad H(T) - H(T|A)$$

Which in words is the difference between the entropy of the parent, $H(T)$, and the total entropy in the left and right leaf nodes respectively, $H(T|A)$. This is the generalized form of information gain, but in Adaboost this form has to be modified to account for the weighted sampling. 

## Entropy for multiclass
As a result of introducing the concept of weighted sampling, we use the concept of **scores** to help compute entropy. The `confidence_score()` function computes the score of a given class, which is the summed weights $w_i$ of the given class `i`, divided by the total weight for all classes. See below.

```
def confidence_score(self, y, w):
    w_total = np.sum(w)
    w_class = np.array([np.sum(w[y==c]) for c in self.n_classes])
    conf_score = w_class / (w_total+1e-10)     
    return conf_score
```
The small value in the total weight is used to prevent division by zero, which is possible if the current weak learner fits the data really well.


# Weighted error rate
The error, $\text{err}^{(m)}$, is a float value that describes the proportion of misclassifications against the total number of classifications. This is an important aspect of Adaboosting, as it is used to ensure that the next weak learner tries to minimize this error. The error is given by:

$$\frac{\sum_{i=1} w_i \cdot I(y_i \neq T^{(m)}(x_i))}{\sum_{i=1}w_i}, \quad \forall i=1, \dots, n$$

However, in practice we generalize this to matrices. Thus we get the following:

$$\text{err}^{(m)} = \frac{w \cdot I(y \neq T^{(m)}(x))}{\sum_{i=1} w_i}, \quad \forall i=1, \dots, n$$

This equation may look very complex, but it will become easier once we break it down.
> $I\left(y_i \neq T^{(m)}(x)\right)$, is called an **indicator function**, it returns 0 for correctly classified samples, and 1 otherwise
>
> $y \neq T^{(m)}(x)$, is a condition that checks whether the y-label and its corresponding prediction $T^{(m)}(x)$ are equal or not
>
> $w$ is the weight matrix, with a shape of $\left(n_ samples,\right)$
>
> $y$ is the matrix containing the labels (responses), with a shape of $\left(n_ samples,\right)$


# Importance weight

# Adaboost in simple steps

# Results

# Conclusion
