**DATASET FROM:** \
https://www.kaggle.com/datasets/himanshunakrani/naive-bayes-classification-data (Univariate Gaussian Naive Bayes) \
https://www.kaggle.com/datasets/karthickveerakumar/spam-filter/data (Multinomial Naive Bayes)

# About the datasets
The datasets we will be working with in this project is "diabetes.csv" and "emails.csv". We will be using the Gaussian Naive Bayes for classifying pasients with diabetes, and those without. The latter (Multinomial Naive Bayes) will be used to classify whether a email should be considered spam or "ham" (not being spam). 

## Dataset 1: "Diabetes.csv"
Represented by a Dataframe object / or Numpy array of shape $\left(995, 3\right)$.
> glucose: int
> bloodpressure: int
> diabetes: int

Where `glucose` and `bloodpressure` are the predictors in the dataset, and `diabetes` being the response variable. 

## Dataset 2: "emails.csv"
Represented by a Dataframe object / or Numpy array of shape $\left(5728, 2\right)$
> text: str
> spam: int

Where the `text` variable is the predictor in the form of sentences (str), and `spam` being the response variable.
