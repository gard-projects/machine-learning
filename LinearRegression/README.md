This project is about learning the mathematics and statistics behind linear regression. To make it more challenging I extended this idea of regression to N dimensions using matrices. Before we begin I encourage you as the reader to have some understanding of what vectors, differentition, and matrices are, as it will help you in understanding the theory and formulas given below.

The dataset we are working with is of the name "l1_test.csv" and "l1_train.csv" which can be found in the dataset folder in my Github repository. The dataset contain two types of variables:

**Independent variable** (also known as predictor, regressor or input) : a variable chosen by the designer which can be adjusted to represent any value, typically placed on the X-axis

**Dependent variable** (also known as response) : the value of this variable changes depending on the value of the predictor(s), this is what we measure as a result of changing the independent variable

# Understanding the dataset
Before we begin it is useful and necessary to understand the dataset you are working with. As a result of this we introduce two new variables which will be important in the code:

**n** : number of samples (observations or data points)

**m** : number of predictors (independent variables)

If we take a look at the dataset for our training set we note the following: \
$n =  696 - 1 = 695$ (we decrease by one to not account for the header label) \
$m = 2 - 1 = 1$ (there is only one dependent variable, y) 

I encourage you as the reader to calculate the respective size (m and n) for the testing set for some practice.

# Extracting data from CSV file
Now that we know the number of independent variables, we can use the variable **m** to extract data from the CSV file into a numpy array. 
Why do we use Numpy arrays? Simply because we can do operations on multiple elements at once (using matrices) instead of doing it one element at a time. This allows for more efficient computation and parallelization. 
To extract data from CSV file we use a Python library called "Pandas". This library provides a function called `read_csv(...)` which takes the file location as an argument. 
Pairing this function with the `to_numpy()` function allows us to put the data into numpy arrays, see code below.

```
training_set = pd.read_csv('./dataset/l1_train.csv').to_numpy()
testing_set = pd.read_csv('./dataset/l1_test.csv').to_numpy()
```
