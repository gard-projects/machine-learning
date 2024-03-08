This project is about learning the mathematics and statistics behind linear regression. To make it more challenging I extended this idea of regression to N dimensions using matrices. 

The dataset that is used in this project can be found here #TODO
We use two sets for the model; training-set for learning the parameters of the network, and a testing set for estimating the true responses of the network.

# (1) Fetching data from csv files
In order to perform operations on data we need to extract the data from the csv files and store it in numpy arrays. We can use Pandas library to achieve this.

```
 training_set = pd.read_csv('./dataset/l1_train.csv'). to_numpy()
testing_set = pd.read_csv('./dataset/l1_test.csv').to_numpy()
```

# (2) Creating the regression model class
This class represents our multivariable model. It takes in two parameters, **epochs** and **learning_rate** $$\alpha$$. 

>Epoch: the number of iterations in the training loop
>
>Learning rate: a step size used in adjusting the weights of the model

The object of this class is used to call the `fit()` function, which acts as our main method for running the model. 

# (3) Standardization and design matrices
Standardization is a very important step, as it allows the gradient to converge faster towards the minimum of the loss function. Additionally it helps in providing better plots, as the estimated weights can become very large (negative or positive) due to the gradient. 

To standardize the dataset we subtract the matrix by the its mean and divide it by its standard deviation. Numpy automatically performs the broadcasting (of the scalars), so you do not need to convert these metrics into matrices. See the code below.

```
def standardize(self, X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X
```

Another important aspect is the use of a **design matrix**. A design matrix is very similar to the features matrix ($$X_{features}$$), but includes an additional column for the bias term of the hypothesis function $$h_{\theta}(x)$$. This matrix makes the equation for the hypothesis function easier, and more compatible with higher dimensional data. 



The code below illustrates how to achieve this.

```
 X = np.c_[np.ones(n), X_features] # Design matrix
```

# (4) Gradient descent and training loop
The equation for the gradient is given by the following equation:



```
def gradient (self, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    y_est = self.predict(X)
    e = y_est - y # Error vector
    g = (1/n) * np.dot(X.T, e)
    return g
```

Where 
