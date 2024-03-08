This project is about learning the mathematics and statistics behind linear regression. To make it more challenging I extended this idea of regression to N dimensions using matrices. 

The dataset that is used in this project can be found here #TODO
We use two sets for the model; training-set for learning the parameters of the network, and a testing set for estimating the true responses of the network.

# (1) Understanding the size of the datasets
In order to understand the structure of the dataset we introduce two new variables **n** and **m**. 

n (int): number of samples
m (int): number of dependent variables

Looking at the training set, we observe the following:
$ n = 695 $
Since we have two variables in total -- one dependent (y) and one independent (x) -- **m** can be found using the following equation:
$ m = 2 - 1 = 1 $

# (2) Fetching data from csv files
In order to perform operations on data we need to extract the data from the csv files and store it in numpy arrays. We can use Pandas library to achieve this.

```
 training_set = pd.read_csv('./dataset/l1_train.csv'). to_numpy()
    testing_set = pd.read_csv('./dataset/l1_test.csv').to_numpy()
```

