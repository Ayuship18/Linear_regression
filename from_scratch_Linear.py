import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from ISLP import load_data

boston = load_data('Boston')

#visualize relationship between lstat and medv
# plt.scatter(boston['lstat'], boston['medv'])
# plt.xlabel('lstat')
# plt.ylabel('medv')
# plt.show()

#prepare data

x_train = (boston['lstat'].values - np.mean(boston['lstat'].values)) / np.std(boston['lstat'].values)
y_train = (boston['medv'].values - np.mean(boston['medv'].values)) / np.std(boston['medv'].values)


print(x_train, y_train)

#cost function

def cost_func(x, y, w, b):
    m = len(x)
    cost = 0

    for i in range(m):
        f = w * x[i] + b
        cost += (f - y[i]) ** 2
    total_cost = cost / (2 * m)
    return total_cost

#gradient descent

def gradient_func(x, y, w, b):
    m = len(x)
    dc_dw = 0
    dc_db = 0

    for i in range(m):
        f = w * x[i] + b
        dc_dw += (f - y[i]) * x[i]
        dc_db += (f - y[i])

    dc_dw = dc_dw / m
    dc_db = dc_db / m

    return dc_dw, dc_db

#get optimal w and b


def gradient_descent(x, y, alpha, iterations):
    #start with random w and b (generally 0)
    w = 0
    b = 0

    for i in range(iterations):
        dc_dw, dc_db = gradient_func(x, y, w, b)

        w = w -alpha * dc_dw
        b = b - alpha * dc_db

        print(f"Iteration {i+1}: Cost {cost_func(x, y, w, b)}")

    return w, b

#training

alpha = 0.0001
iterations = 10000
final_w, final_b = gradient_descent(x_train, y_train, alpha, iterations)
print(f"Final weight: {final_w}, Final bias: {final_b}")

#unscalling the coefficients
std_x = np.std(boston['lstat'].values)
std_y = np.std(boston['medv'].values)
mean_x = np.mean(boston['lstat'].values)
mean_y = np.mean(boston['medv'].values)
final_w = final_w * (std_y / std_x)
final_b = mean_y - final_w * mean_x + final_b * std_y
print(f"Unscalled weight: {final_w}, Unscalled bias: {final_b}")


