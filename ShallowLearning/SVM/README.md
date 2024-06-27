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

## The Dual Problem and the Kernel Trick
In optimization theory, there exists a corresponding problem to the primal problem known as the **dual problem**. Solving the dual problem can offer different computational or theoretical advantages, and in some cases, it might be less computationally intensive. A part of our implementation focuses on the dual problem because it allows us to employ the kernel trick. This approach is particularly beneficial in scenarios like support vector machines, where it enables the handling of high-dimensional feature spaces more efficiently. The dual problem for primal problem specified above is given as:

$$max_{\alpha} \hspace{0.1cm} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}\left(x_{i} \boldsymbol\cdot x_{j}\right)$$

$$\text{subject to} \quad 0 \leq \alpha_{i} \leq C \quad \forall i \quad \text{and} \quad \sum_{i=1}^{n} \alpha_{i}y_{i} = 0$$

The interesting part about this equation is the **dot product** between $x_i$ and $x_j$, since we can replace this with a kernel matrix, called the **Gram matrix**.

$$G_{ij} = \langle x_{i}, x_{j} \rangle$$

In essence, the Gram matrix stores the dot products between every pair of $x_i$ and $x_j$. This allows for using the kernel trick, since we do not perform the operation to move the points to a higher dimensional space, we only compute the transformed dot products. The code below illustrates one of the kernel functions that is implemented in the SMO class. 
```
def polynomial_kernel(X, Y=None, r=0, d=3, gamma=None):
    '''
    Parameters
    ----------
    X : np.ndarray
        A matrix with n samples and m features, shape (n, m)
    Y : np.ndarray
        A matrix with n samples and m features, shape (n, m)
    r : float
        Coefficient of the polynomial kernel
    d : int
        Degree of the polynomial kernel
    gamma : float
        Coefficient of the polynomial kernel, used for scaling
        
    Returns
    -------
    The polynomial kernel matrix for the given data, shape (n, n)
    '''
    if Y is None:
        Y = X
        
    K = np.dot(X, Y.T)
    if gamma is None:
        gamma = 1 / (X.shape[1]*np.var(X))
    
    return (gamma * K + r)**d
``` 

# Sequential Minimal Optimization (SMO)
Is an algorithm that solves the quadratic programming (QP) problem that arises during training in SVM. I noticed that this approach is not as efficient as the QP-solver Sklearn uses on this dataset, thus it take a bit longer to train the model. I chose to create a separate class for the SMO, since it made it more convenient to handle the large amount of code. To start of, the main function of this class is `sequential_minimal_optimization(...)`, and it is invoked by the `fit(...)` method in the SVM class.

This function uses hyperparameters to find the alphas $\alpha_i$. Lets cover them.
> **max_iters** - the number of iterations for updating the $\alpha_i$
>
> **tol** - a tolerance level that decides what we consider a optimal solution
>
> **C** - the regularization parameter, used to reduce the misclassifications by choosing suitable $\alpha_i$ that reduces the amount of error

Other useful variables used are:
> **kernel_cache** - used to store the kernel matrix, K, returned by the chosen kernel function. It $\underline{significantly improves performance}, due to less computation!
>
> **errors** - uses the result of `compute_error(...)` function to residual between computed score and actual score (the actual score being $y_i \in \lbrace -1, 1 \rbrace$)

## Compilation steps in the SMO algorithm
For each iteration we check the **Karush-Kuhn-Tucker (KKT)** conditions. Which are first derivative tests for a solution in non-linear programming to be optimal. This theorem is also known as **saddle-point theorem**. We have the following.

### 1. Check two conditions to determine if the given $\alpha_i$ should be optimized 
🟢 $y_i = -1 \quad \Rightarrow \quad y_i \cdot E_i < -tol \quad \text{and} \quad \alpha_{i} < C$ \
🔵 $y_i = 1 \quad \Rightarrow \quad y_i \cdot E_i > tol \quad \text{and} \quad \alpha_{i} > 0$ 

This checks the following conditions:


➡️ **Primal feasibility** \
The requirement in context of SVM means that all data points must be on  or outside the margin boundary according to their class labels. For any data point $x_i$ with label $y_i$ the condition is: \
$$y_{i} \cdot \left(\text{decision function}\right) \geq 1$$

Thus if $y_i = 1$ the point should be on or above the boundary, and if $y_i = -1$ the point show be on or below the boundary.


➡️ **Dual feasibility** \
Relates to the constraints on the Lagrange multipliers $\alpha_i$. Each $\alpha_i$ corresponds to a training example $x_i$. These coefficients are used to optimize the margin width.

* Each $\alpha_i$ must be non-negative
* Each $\alpha_i$ must not exceed the upper bound **C** (the regularization parameter)


➡️ **Complementary slackness** \
This property ensures that the Lagrange multipliers are used efficiently. It states the following:

* If $\alpha_i > 0$, then the corresponding data point $x_i$ lies exactly on the margin boundary, thus $x_i$ is a **support vector**
* If $\alpha_i = 0$, the data point $x_i$ is either correctly classified beyond the margin, or potentially misclassified or within the margin (in the case of Soft Margin SVM)


### 2. Choose the second alpha, $\alpha_j$
We choose a random alpha to get a more representative solution to the optimization problem without any potential bias.
```
j = np.random.randint(low=0, high=n-1)
```


# Sources
Singh, N. (2023). Soft Margin SVM / Support Vector Classifier (SVC) [Graph]. https://pub.aimind.so/soft-margin-svm-exploring-slack-variables-the-c-parameter-and-flexibility-1555f4834ecc
