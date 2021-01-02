import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


import statsmodels.api as sm
X_train=np.append(arr=np.ones((40,1)).astype(int), values=X_train, axis=1)
X_test=np.append(arr=np.ones((10,1)).astype(int), values=X_test, axis=1)

X_opt = np.array(X_train[:, [0,  3, 5]], dtype=float)
X_opt_1 = np.array(X_test[:, [0,  3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
dataX=regressor_OLS.predict(X_opt_1)
print(regressor_OLS.summary())


fig = sm.graphics.influence_plot(regressor_OLS, criterion="cooks")
fig.tight_layout(pad=1.0)

