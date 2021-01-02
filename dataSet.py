import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#open csv files

data=pd.read_csv("data.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,3].values



#fill missing data
from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
#imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.fit_transform(x[:,1:3])



#convert string data to numbers
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder=LabelEncoder();
x[:,0]=labelEncoder.fit_transform(x[:,0])
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)
y=labelEncoder.fit_transform(y)

#slipt data 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
St_salar=StandardScaler()
X_train=St_salar.fit_transform(X_train)
X_test=St_salar.transform(X_test)





