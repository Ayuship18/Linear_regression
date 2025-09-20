import numpy as np
from sklearn.datasets import make_regression

#creating dataset
X, y = make_regression(n_samples=100, n_features=1, 
                       noise=10, random_state=42)

def linear_regression_normal_equation(X, y):
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)

    theta = np.linalg.solve(X_transpose_X, X_transpose_y)
    return theta

# Add a column of ones to X for the intercept term
X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

theta = linear_regression_normal_equation(X_with_intercept, y)
if theta is not None:
    print(theta)
else:
    print("Unable to compute theta. The matrix X_transpose_X is singular.")