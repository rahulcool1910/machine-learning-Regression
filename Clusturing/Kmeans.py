import pandas as ps
import numpy as np
import matplotlib.pyplot as plt

dataSet=pd.read_csv("mall.csv")
X=dataSet.iloc[:,[3,4]].values


from sklearn.cluster import KMeans

Wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    Wcss.append(kmeans.inertia_)
             
plt.plot(range(1,11),Wcss)    

kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
Y_pred=kmeans.fit_predict(X)

plt.scatter(X[Y_pred==0,0], X[Y_pred==0,1], color='red')
plt.scatter(X[Y_pred==1,0], X[Y_pred==1,1], color='blue')
plt.scatter(X[Y_pred==2,0], X[Y_pred==2,1], color='green')
plt.scatter(X[Y_pred==3,0], X[Y_pred==3,1], color='orange')
plt.scatter(X[Y_pred==4,0], X[Y_pred==4,1], color='violet')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,color='magenta')




