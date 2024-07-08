**DATASET FROM**: \
 https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression

 # About the dataset
 The dataset has the following columns with a shape of $\left(4238, 16\right)$. They primary objective is to predict whether a patient is likely to developing coronary heart disease (CHD) within the next 10 years, given the provided features. Thus `TenYearCHD` is the response variable.
 > male: int
 >
 > age: float
 >
 > education: int
 >
 > currentSmoker: int
 >
 > cigsPerDay: float
 >
 > BPMeds: int
 >
 > prevalentStroke: int
 >
 > prevalentHyp: int
 >
 > diabetes: int
 >
 > totChol: float
 >
 > sysBP: float
 >
 > diaBP: float
 >
 > BMI: float
 >
 > heartRate: float
 >
 > glucose: float
 >
 > TenYearCHD: int

&nbsp;

# Shapes of numpy arrays
> **x** (the features of a patient): $\left(4238, 15\right)$ 
> 
> **y** (the targets): $\left(4238,\right)$ 
>
> **y\_pred** (evaluated using sigmoid function $\sigma(z)$ ): $\left(4238,\right)$ 
> 
> **X** (the design matrix): $\left(4238, 16\right)$ 
> 
> **w** (the weights, including the bias term `b`): $\left(16,\right)$ 
> 
> **z** (the logits): $\left(4238,\right)$ 
> 
> **grad** (the gradient vector): $\left(16,\right)$

&nbsp;

# The Basics of Logistic Regression
This type of model is quite similar to **linear regression** in many ways, but deals with classifications as opposed to predicting continuous values. Linear regression uses a linear function to make predictions, while logistic regression uses a S-shaped curve, which is actually the **sigmoid function $\sigma(z)$**. The image by Toprak (2020) illustrates the behaviour of the sigmoid function, see below.
![sigmoid_function](../images/sigmoid_function.png)

The advantage of using a sigmoid function is that the value range (y-axis) is restricted to 0 and 1. This makes sense intuitively as we cannot have probabilities exceeding 1, and below 0. The steps for this estimator can be seen below.

1. Data preparation, perform normalization on features, split data into training and testing sets respectively
2. Initialize the parameters `w` randomly
3. Define the Sigmoid function $\sigma(z)$
$$\sigma(z) = \frac{1}{1+e^{-z}$$
5. Define the logit function
$$z = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n \quad \eq X \cdot w$$

# Sources
Toprak, M (2020). Activation Functions for Deep Learning [Image]. https://medium.com/@toprak.mhmt/activation-functions-for-deep-learning-13d8b9b20e
