import numpy as np
from sklearn.linear_model import LinearRegression

# Regressors (input x)
x = np.array([10,20,33,88,22,33,50,70]).reshape((-1,1))
# Response (output y)
y = np.array([20,30,40,50,60,70,80,90])


# Create a model
model = LinearRegression()
# Calculate optimal values of weights (fits the model)
model.fit(x,y)

# Get coefficient of determination (R^2)
r_score = model.score(x,y)
print(f"R^2: {r_score}")
# By convention, a underscore '_' is used to indicate estimated parameters
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_}")

# Get predictions 
y_pred = model.predict(x) 
print(f"Predictions: {y_pred}")
# Alternatively...
y_pred_alt = model.intercept_ + model.coef_ * x
print(f"Predictions (alternative): {y_pred_alt}")
nmb_features = model.n_features_in_
print(f"Number of features: {nmb_features}")

