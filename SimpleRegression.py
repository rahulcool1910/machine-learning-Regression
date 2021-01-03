import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data=pd.read_csv("Salary_Data.csv")

X=data.iloc[:, :-1].values
Y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


'''from sklearn.preprocessing import StandardScaler

SS=StandardScaler();
X_train=SS.fit_transform(X_train)
X_test=SS.fit_transform(X_test)
Y_train=SS.fit_transform(X_train)
Y_test=SS.fit_transform(X_test)'''



from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train,Y_train)


Y_pred=Regressor.predict(X_train)


plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,Y_pred,color='blue')
plt.title("Salary Vs Experiance")
plt.xlabel("Salary")
plt.ylabel("Experiace")
plt.show()


 