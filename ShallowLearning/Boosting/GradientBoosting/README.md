**DATASET**: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

&nbsp;

# The structure of the dataset
This dataset features a response variable `stroke` which has a binary outcome, 0 or 1. The goal in this project is classify patients that may develop stroke, given a set of features, **X**.
> **id**, type=int, predictor
> 
> **gender**, type=str, predictor variable
> 
> **age**, type=int, predictor variable
>
> **hypertension**, type=int, predictor variable
>
> **heart_disease**, type=int, predictor variable
>
> **ever_married**, type=str, predictor variable
>
> **work_type**, type=str, predictor variable
>
> **residence_type**, type=str, predictor variable
>
> **avg_glucose_level**, type=float, predictor variable
>
> **bmi**, type=float, predictor variable
>
> **smoking_status**, type=str, predictor variable
>
> **stroke**, type=int, **response variable**

&nbsp;

# Data preprocessing
In order to be able to train the model all features need to be numerical, which you can achieve through **encoding**. There exists many different encoding algorithms, each with their own advantages and downsides. I chose to use **one-hot encoding**, as the number of unique categories for each of the respective categorical features are relatively low. 

Additionally, some of the columns in the dataset contains `NaN` values, or rather missing values. This is quite common in datasets, as some participants may choose to not enclose personal information. To account for this problem we use `SimpleImputer()` function from Sklearn, which is configured to replace the missing values for a given feature with the median of this feature. To summarize we apply three transformations on the dataset using Sklearn's pipeline object. This object is quite useful, as it allows us to apply a series of transformations to the data in a sequential manner.
```
cat_features = [1, 5, 6, 7, 10]
num_features = [i for i in range(X.shape[1]) if i not in cat_features]
    
preprocessor = ColumnTransformer([
  ('num', Pipeline([
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())
  ]), num_features),
  ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X = preprocessor.fit_transform(data)
```

It is usually reasonable to assume a normal distribution of the variables. Not only does it make computation simpler, but it decreases the total amount of time needed to train the model. We use `StandardScaler()` from Sklearn to standardize a feature. There is also a large imbalance in the dataset between the number of participants that have experienced a stroke, and does who have not. This is a problem, as it may introduce more bias making the model perform better on those who have not experienced a stroke as opposed to those that have. To solve this issue we introduce the concept of **SMOTE** (from imblearn), which handles the undersampling of patients with strokes (label = 1). Additionally we use a **stratified sampling method** to randomly select patients from both groups, thus reducing the heavy imbalance of the class distribution.

```
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

To train each of the models we again use the pipeline object. This will first apply the appropriate preprocessing steps, then it will train the model. 
```
 c_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoost())
    ])

    sk_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier())
    ])
```

&nbsp;

# Negative log loss function
This **objective function** is very common in classification algorithms like **Logistic regression** and also in **Gradient boosting**. It is defined in the following way:

$$L(y_i, p_i) = \quad - \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right]$$

Where $y_i$ is the label of the current sample, and $p_i$ is the predicted label of the current weak learner. We can calculate $p_i$ by using the **Sigmoid function**, $\sigma(y_i)$.

$$p_i = \frac{1}{1 + e^{(-y_i)}}$$

&nbsp;

# The concept of function space
What makes GradientBoosting (and Adaboost for instance) different from other models is that this algorithm works directly with the **function space**, not the analytical expression of the model. Linear regression is an example of a model that uses a analytical expression, in which the goal is to optimize the parameters making up the equation for the model. However GradientBoosting is different, as we are not trying to optimize any parameters, but rather the output of the sequential weak learners. Thus the final model in this boosting algorithm is the weighted sum of the trained weak learners, rather than a single analytical expression. More formally:

$$F_m(x) = F_{m-1}(x) + \gamma_{m} h_{m}(x)$$

As opposed to linear regression

$$F(x) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$$

In the context of machine learning, **function space** is the set of all possible functions that a particular model can learn. For gradient boosting the function space is the set of all possible weighted sums of decision trees (weak learners). The function space is very convenient, as it allows our model to deal with complex non-linear relationships in the data. On the other hand linear regression works in a function space that is more constrained, that is all possible linear equations. The main takeaway is that each weak learner in Gradient boosting does not have an analytical form, we are working directly with the outputs (predictions) of each weak learner, in fact this is what we are adding up! Thus, the final strong learner is essentially just the sum of all the predictions of the weak learners. 

&nbsp;

# Functional gradient descent
Gradient boosting is also different in how it computes gradients and uses them. Usually when we compute the gradient of a loss function $L(y, p)$, we compute it with respect to the objective function's parameters. An example of this is in linear regression, where we compute the gradient with respect to the betas, $\beta$. However in graient boosting we compute the gradient with respect to a function, namely the **predict function** which is defined as $F_m(x_i)$. We call this type of gradient, **functional gradient**. This is interesting, as parameters (a.k.a weights) are typically "fixed", while functions represent a large range of values (typically an infinite range). 

So how do we derive the gradient in such a case? From the definition of negative log loss function

$$L(y_i, p_i) = \quad - \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right]$$

$$\frac{\partial L(y, p)}{\partial \hat{y}} = \quad - \left[y_i \cdot \frac{1}{p_i} \cdot \frac{e^{-\hat{y}}}{\left(1 + e^{-\hat{y}}\right)^{2}} \quad - \quad (1 - y) \cdot \frac{\left(1 + e^{-\hat{y}}\right)}{e^{-\hat{y}}} \cdot \frac{e^{-\hat{y}}}{\left(1 + e^{-\hat{y}}\right)^{2}}\right]$$

Simplifying this further, gives us

$$\frac{\partial L(y, p)}{\partial \hat{y}} = \quad - \left[y_i \cdot \frac{e^{-\hat{y}}}{\left(1 + e^{-\hat{y}}\right)} - (1-y) \cdot \frac{1}{\left(1 + e^{-\hat{y}}\right)} \right]$$

Where we have the following:

$$\left(1 - p\right) = \quad \frac{e^{-\hat{y}}}{\left(1 + e^{-\hat{y}}\right)}$$

$$p = \quad \frac{1}{\left(1 + e^{-\hat{y}}\right)}$$

Finally we have...

$$\frac{\partial L(y, p)}{\partial \hat{y}} = \quad - (y - p)$$

This is the gradient we will be using to compute the pseudo-residuals (more on this in the next section).
&nbsp;

# Pseudo residuals
A pseudo-residual is an approximation of the negative gradient of a loss function, with respect to the model's predictions. The key concept here is that gradient boosting tries to minimize a loss function iteratively, by computing the pseudo-residual in each iteration and training the next weak learner on the pseudo-residual. In other words, the pseudo-residuals become the target variable for the next weak learner. See the code below.

```
y_pred = 1 / (1+np.exp(-F))
grad = -(y - y_pred) # Pseudo-residual
            
h_m = DecisionTreeRegressor().fit(X, grad)
```
Notice how $h_m$ (which is the (m+1) weak learner) is trained on both the features matrix, $X$, and the **pesudo-residuals**! This allows us converge towards a minimum in an iterative manner using weak learners. Note that this differs quite a lot from standard regression, as we only train one model and minimize the loss function using this model. However, in this algorithm we constantly minimize the loss function with a model that continues to change (in each iteration), making the minimization process different.
&nbsp;

# The step by step process
1. Initialize $F_0(x)$ as the proportion of true positives (strokes) up against the total number of samples
```
F = np.full((len(y),), np.log(p/ (1-p))) 
```
2. Train $m$ weak learners by using a for loop
3. Compute the pseudo-residuals

$$r_{i, m} = \quad - \left. \frac{\partial L(y, F(x)}{\partial F(x)} \right\rvert_{F(x)= F\_{m-1}(x)}$$

4. Fit a weak learner closed under scaling $h_m(x)$ to the pseudo-residual. Train it using a training set in the form of: $`\{ X, r_m \}`$. Which equates to this line of code:
```
h_m = DecisionTreeRegressor().fit(X, grad)
```

&nbsp;

# Results and conclusion
