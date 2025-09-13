# Linear-Regression-from-scratch
##🔍 Description

This project demonstrates my understanding of linear regression by building the model in two different ways:

From scratch — implementing gradient descent manually in Python.

Using prebuilt functions from the ISLP.models package.

>The aim is to show that both approaches arrive at the same regression line, while highlighting the practical considerations (such as scaling features and interpreting coefficients) that are critical when applying regression in real datasets.

##📊 Dataset

I use the Boston Housing dataset (ISLP.load_data("Boston")), focusing on:

*lstat: % of population with lower socioeconomic status (predictor)*

*medv: Median value of owner-occupied homes in $1000s (response)*

##🧮 Equations

f(x)=wx+b
where:

w = weight (slope)

b = bias (intercept)

**Cost function (Mean squared error)**

```math
J(w,b)=1/2m​\sum_{i=1}^n(f(x_i​)−y_i​)^2
```

**Gradients**

```math
∂J​/∂w = 1/m \sum_{i=1}^n (f(x_i)-y_i)^2*x_i
```
```math
∂J​/∂b = 1/m \sum_{i=1}^n (f(x_i)-y_i)^2
```

**Gradient descent update rule**

```math
w = w - α∂J/∂w​
```
```math
b = b - α∂b/∂J​
```
*where α is the learning rate*

⚖️ Scaling and Unscaling
Why scale?

Gradient descent works better when features are on similar scales. Without scaling, the optimization landscape looks like a long narrow valley, causing gradient descent to zig-zag or diverge. With scaling, the valley becomes a round bowl, making convergence faster and more stable.

Unscaling coefficients

To compare with the prebuilt model (which uses unscaled data), we convert coefficients back:

```math
w_(orig)​ = w⋅σx​/σy​​
```
```math
b_(orig)​=μy​+σy​⋅b−w_(orig)​⋅μx​
```
✨ Key Learnings

Gradient descent requires careful choice of learning rate; too high leads to divergence, too low converges slowly.

Scaling is essential for stable optimization, but interpreting results requires unscaling back to the original units.

Writing it from scratch deepened my intuition for how linear regression works under the hood.

🚀 Future Work

Extend to multiple linear regression (more predictors).

Try regularization (Ridge, Lasso).

Explore logistic regression for classification tasks.


