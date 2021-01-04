import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataSet=pd.read_csv("Social_Network_Ads.csv")
X=dataSet.iloc[:,[2,3]].values
y=dataSet.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X_train=st.fit_transform(X_train)
X_test=st.transform(X_test)


from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)












