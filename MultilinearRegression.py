import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#open csv files
data=pd.read_csv("50_Startups.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values




from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer


labelEncoder=LabelEncoder();
x[:,3]=labelEncoder.fit_transform(x[:,3])

ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')

x = ct.fit_transform(x)

x=x[:,1:]


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train,Y_train)

Y_pred=Regressor.predict(X_test)


