import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataSet=pd.read_csv("Social_Network_Ads.csv")
X=dataSet.iloc[:,[2,3]].values
Y=dataSet.iloc[:,4].values



from sklearn.preprocessing import StandardScaler

SS=StandardScaler()

X=SS.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
 



from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)


from sklearn.metrics import  confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)



from matplotlib.colors import ListedColormap
X_set,Y_set=X_train,Y_train
X1,X2=np.meshgrid(np.arange(X_set[ :,0].min()-1,X_set[:,0].max()+1,step=0.01)
                  ,np.arange(X_set[:,1].min()-1,X_set[:,1].max()+1,0.01))
X_pre=np.array([X1.ravel(),X2.ravel()]).T
plt.contourf(X1,X2,classifier.predict(X_pre).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(("red","green")))

