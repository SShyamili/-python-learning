import pandas as pd
import matplotlib.pyplot as plt 
import pandas as pd
 
dataset=pd.read_csv(r"C:\Users\S SHYAMILI\Downloads\emp_sal.csv")

X=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

plt.scatter(X,y,color='green')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("linear regression model(linear regression)")
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

m=lin_reg.coef_
print(m)

c=lin_reg.intercept_
print(c)

lin_model_pred=lin_reg.predict([[6.5]])
lin_model_pred

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=5)
X_poly=poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('polymodel (polynomial regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

poly_model_pred=lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred





from sklearn.svm import SVR
svr_reg=SVR(kernel='poly',degree=4,gamma='auto')
svr_reg.fit(X,y)

svr_model_pred=svr_reg.predict([[6.5]])
svr_model_pred



from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor(n_neighbors=4,weights='uniform')
knn_reg.fit(X,y)

knn_model_pred=knn_reg.predict([[6.5]])
knn_model_pred


from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(criterion="squared_error")
dt_reg.fit(X,y)

dt_model_pred=dt_reg.predict([[6.5]])
dt_model_pred

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(random_state=0)
rf_reg.fit(X,y)

rf_model_pred=rf_reg.predict([[6.5]])
rf_model_pred


