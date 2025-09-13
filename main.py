import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import statsmodels.api as sm 
from ISLP import load_data
from ISLP.models import (ModelSpec as ms, summarize, poly)

#load data
#linear regression with x as lstat and y as medv

boston = load_data('boston')

#creating dataframe
X = pd.DataFrame({'intercept': np.ones(len(boston['lstat'])),
                  'lstat': boston['lstat']})

y = boston['medv']

#fit model
model = sm.OLS(y,X) #specify model
results = model.fit() #fits the model
print(summarize(results)) #summarize results

'''

              coef  std err       t  P>|t|
intercept  34.5538    0.563  61.415    0.0
lstat      -0.9500    0.039 -24.528    0.0
'''