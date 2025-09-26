import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
'''
Honeybees are currently facing a fragile situation.
You’ve may have heard or read reports on a decline in their populations due to multiple factors.
The goal of this algorithm is to explore this decline and examine how past patterns can help forecast the future of honeybees.
Data on honey production in the United States has 8 features and 628 samples: 
1. State, 2. Number of colonies, 3. Yield per colony, 4. Total production, 5. Stocks held by producers, 6. Price per pound, 7. Production value, and 8. Year. 

The National Agricultural Statistics Service (NASS) is the primary data reporting body for the US Department of Agriculture (USDA). 
NASS's mission is to "provide timely, accurate, and useful statistics in service to U.S. agriculture". 
From datasets to census surveys, their data covers virtually all aspects of U.S. agriculture. Honey production is one of the datasets offered.
https://www.nass.usda.gov/About_NASS/index.php 
https://usda.library.cornell.edu/MannUsda/viewDocumentInfo.do
'''

df = pd.read_csv("honeyproduction.csv")
#Data exploration
print(df.columns)
print(df.head)
print(df.shape)

# Data transformation 
# Groups total production for all states (5,105,093 in 1998) per year and gets the mean
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
print(prod_per_year)

x = prod_per_year['year']
x = x.values.reshape(-1, 1)
y = prod_per_year['totalprod']
plt.scatter(x, y)
plt.show()

regr = linear_model.LinearRegression()
regr.fit(x,y)
print(regr.coef_)
print(regr.intercept_)

y_predict = regr.predict(x)
plt.plot(x, y_predict)
plt.show()

# Prediction
nums = np.array(range(1, 11))
x_future = np.array(range(2013, 2025))

# Reshapes to column instead of row
x_future = x_future.reshape(-1, 1)
print(x_future)
future_predict = regr.predict(x_future)
print(future_predict)
plt.plot(x_future, future_predict)
plt.show()

#2050 prediction is 186,545
prediction = (regr.coef_ * 2050) + regr.intercept_
print(prediction)
