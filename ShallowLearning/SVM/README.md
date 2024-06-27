Dataset from: https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset?resource=download

This project is rather unique, as it has involved implementing both the estimator, but also a form of quadratic programming solver.
Support Vector Machines are estimators, or rather models that aim to classify new data points into either of two groups. Our dataset covers raisins grown in Turkey, and the goal is to predict whether a new raisin is of type "Kecimen" or "Besni". Thus we are interested in understanding how well our custom model can classify a raisin as a Kecimen or Besni type. 

# Structure of the raisin dataset
The shape of the dataset is (900, 8). Meaning a sample axis of 900 dimensions, and a features axis with 8 dimensions. There are the following features:

> **Area** , type=float
> 
> **MajorAxisLength**, type=float
> 
> **MinorAxisLength**, type=float
> 
> **Eccentricity**, type=float
> 
> **ConvexArea**, type=float
> 
> **Extent**, type=float
> 
> **Perimeter**, type=float
> 
> **Class**, type=string 

Thus there are in total 7 quantitative variables (only continuous), and 1 categorical variable (being the response variable).

# Introduction to SVM, and what to solve
Support Vector Machine (SVM) is a type of algorithm that aims to classify new points by using a **decision boundary**. In simple terms, a decision boundary is a N-1 dimensional object (a hyperplane in mathematical terms) which existence is to cleanly separate two groups in some N-space. A SVM can be implemented in two ways, using a **Hard Margin Classifier** and a **Soft Margin Classifier**. A the former requires that the data is **linearly separable**, i.e. that you separate the data without any of the two groups touching each other. This approach is typically not possible to implement in practice, as there is often noise in data leading to messy patterns. The latter (Soft Margin Classifier) is an approach that is suitable in a real-world environment as it allows for misclassifications which is a core problem of machine learning (bias-variance tradeoff). The image below by Singh (2023) illustrates the difference between these two implementations of SVM graphically.

![Image of both Soft Margin Classifier and Hard Margin Classifier](../images/svm_types.png)


So how exactly is the data separated? By maximizing the distance between **support vectors** and the **decision boundary**. A support vector is a data point (or individual) which lies within the margin, or on the margin. These are the data points that matter the most when optimizing the decision boundary. The Hard Margin Classifier focuses only on maximizing distance, while the Soft Margin Classifier focuses on both maximizing the margin, **but also** minimizing the misclassifications through a **hyperparameter** called the **regularization** parameter `C`. A hyperparameter is a parameter that is not learned by the model, but has to be set manually by a human. From the image we have the following equations:


$(1)\quad \vec{w} \cdot \vec{x} + b = 1$

$(2)\quad \vec{w} \cdot \vec{x} + b = 0$

$(3)\quad \vec{w} \cdot \vec{x} + b = -1$

Therefore the main goal is to find the weight vector **w** and the bias term **b**. We can find these variables by solving the primal problem for soft margin SVM given by: \
$$min_{w \hspace{0.05cm},b \hspace{0.05cm},\xi}\hspace{0.1cm} \|| w^{2} \|| + C \sum_{i=1}^n \xi$$ 
$$s.t. \quad y_{i}\left(w^{T}x_{i} + b\right) \geq 1 - \xi_{i} \quad \forall i = 1, \dots, n$$

From the equation we have the following: 

$y_{i}\left(w^{T}x_{i} + b\right)$ is the decision function, used to compute the distance between each point and the hyperplane (Support Vector Classifier) 

$w$ is the weight vector, it represents all the weights of the hyperplane 

Sum of all the penalties $\xi_{i}$ given by:
$$\sum_{i=1}^n \xi = \xi_{1} + \xi_{2} + \dots + \xi_{n}$$

## Defining the meaning of penalty
The penalty, $\xi$, in the context of SVM is used to handle misclassifications, and points that fall within the margin. Each penalty calculated by using the **hinge loss function**:

$$\ell(x) = max(0, 1 - y_{i}\left(w \cdot x - b \right))$$

There are 3 possible cases that can occur when deciding on the penalty of a point.

🟧 **Case 1: Correctly classified and outside the margin**
If $y_{i}\left(w \cdot x - b \right) > 1$, the output of the classification function is outside the boundary of the margin

🟪 **Case 2: On the margin or misclassified**
If $y_{i}\left(w \cdot x - b \right) < 1$, either the point is on the wrong side of the margin but correctly classified (between the decision boundary and the margin), or the point is misclassified

🟥 **Case 3: Exactly on the decision boundary**
If $y_{i}\left(w \cdot x - b \right) = 0$ the point lies exactly on the decision boundary.

## The Dual Problem
In optimization theory, there exists a corresponding problem to the primal problem known as the **dual problem**. Solving the dual problem can offer different computational or theoretical advantages, and in some cases, it might be less computationally intensive. A part of our implementation focuses on the dual problem because it allows us to employ the kernel trick. This approach is particularly beneficial in scenarios like support vector machines, where it enables the handling of high-dimensional feature spaces more efficiently. The dual problem for primal problem specified above is given as:

$$max_{\alpha} \hspace{0.1cm} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}\left(x_{i} \boldsymbol\cdot x_{j}\right)$$

$$\text{subject to} \quad 0 \leq \alpha_{i} \leq C \quad \forall i \quad \text{and} \quad \sum_{i=1}^{n} \alpha_{i}y_{i} = 0$$



The expression above does to things, \
(1) It maximizes the margin through minimizing `w` \
(2) It minimizes misclassifications through the second term using the hyperparameter `C`

It turns out solving the dual problem is better in terms of performance, as typically most of the Lagrange multipliers are zero (non support vectors), and the kernel trick can be utilized. 


# Sources
Singh, N. (2023). Soft Margin SVM / Support Vector Classifier (SVC) [Graph]. https://pub.aimind.so/soft-margin-svm-exploring-slack-variables-the-c-parameter-and-flexibility-1555f4834ecc
