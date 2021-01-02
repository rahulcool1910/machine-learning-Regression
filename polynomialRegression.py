import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



dataset=pd.read_csv("Position_Salaries.csv")


x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values



from sklearn.preprocessing import PolynomialFeatures


poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(x)


from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(x,y)
arr=np.array([6.5,7.6]).reshape(-1,1)

X_grid=np.arange(min(x),max(x),0.1)
X_grid=X_grid.reshape(len(X_grid),1)

lin_reg.fit(X_poly,y)
print(lin_reg.predict(poly.fit_transform(arr)))

plt.scatter(x,y,color='blue')
#plt.plot(X_grid,lin_reg.predict(poly.fit_transform(X_grid)),color='red')

X_check=np.arange(6.5,7.5,1)
X_check=X_check.reshape(len(X_check),1)
plt.scatter(X_check,lin_reg.predict(poly.fit_transform(X_check)),color='green')

