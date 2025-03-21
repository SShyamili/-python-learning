import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset=pd.read_csv(r"C:\Users\S SHYAMILI\OneDrive\Desktop\data science\Salary_Data.csv")
x = dataset.iloc[:, :-1]  
y = dataset.iloc[:, -1] 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


x_train = x_train.values.reshape(-1, 1)

x_test = x_test.values.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test) 


plt.scatter(x_test, y_test, color = 'red')  
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_train, y_train, color = 'red')  
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

coef=print(f"coefficient: {regressor.coef_}")
intercept=print(f"intercept: {regressor.intercept_}")

comparision=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
print(comparision)

ex_12_future_pred=9312*20+26780

ex_12_future_pred


bias=regressor.score(x_train,y_train)
print(bias)
variance=regressor.score(x_test,y_test)
print(variance)

dataset.mean()
dataset["Salary"].mean()
dataset.var()
dataset.std()
dataset["Salary"].std()


from scipy.stats import variation
variation(dataset.values)
dataset.corr()
dataset["Salary"].corr(dataset["YearsExperience"])
dataset.skew()
dataset["Salary"].skew()
dataset.sem()
dataset["Salary"].sem()

import scipy.stats as stats
dataset.apply(stats.zscore)
a=dataset.shape[0]
b=dataset.shape[1]
degree_of_freedom=a-b
print(degree_of_freedom)


y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

r_square=1-(SSR/SST)
r_square


print(regressor)


