import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



dataSet=pd.read_csv("Position_Salaries.csv")
X=dataSet.iloc[:,1:2].values
Y=dataSet.iloc[:,2].values



from sklearn.tree import DecisionTreeRegressor
Regressor=DecisionTreeRegressor(random_state=0)


Regressor.fit(X,Y)



print(Regressor.predict(np.array([6.5]).reshape(-1,1)))
X_grid=np.arange(0,10,0.01).reshape(-1,1)

Y_pred=Regressor.predict(X_grid)

plt.scatter(X,Y,color="red")
plt.plot(X_grid,Y_pred,color='blue')