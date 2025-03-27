import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
dataset=pd.read_csv(r"C:\Users\S SHYAMILI\OneDrive\Desktop\data science\Investment.csv")

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

X=pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

m_slope=regressor.coef_
print(m_slope)

c_inter=regressor.intercept_
print(c_inter)

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,4,5]]

regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

import statsmodels.api as sm
X_opt=X[:,[0,1,2,3]]

regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()





